import io
import os
import json
import base64
import tempfile
import numpy as np

from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError,
)

from flask import Flask, render_template, request, jsonify
from tensorflow import keras

app = Flask(__name__)

MODEL_PATH = "model/emotion_model.h5"
CLASS_MAP_PATH = "model/class_indices.json"
IMG_SIZE = 224
PREVIEW_SIZE = (320, 320)
PREVIEW_BACKGROUND = (248, 250, 252)
FACE_CROP_PAD_RATIO = 0.14

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

DEFAULT_EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

# =========================
# Load model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Emotion model not found at: {MODEL_PATH}")

emotion_model = keras.models.load_model(MODEL_PATH)

if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    EMOTIONS = {int(v): k.capitalize() for k, v in class_indices.items()}
else:
    EMOTIONS = DEFAULT_EMOTIONS

# =========================
# Optional MediaPipe
# =========================
MEDIAPIPE_AVAILABLE = False
face_detector = None

try:
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        MEDIAPIPE_AVAILABLE = True
        print("MediaPipe face detection loaded successfully.")
    else:
        print("MediaPipe installed, but face detection is unavailable.")
except Exception as e:
    print(f"MediaPipe unavailable: {e}")

# =========================
# Optional OpenCV
# =========================
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

haar_cascade = None
if CV2_AVAILABLE:
    try:
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(haar_path):
            haar_cascade = cv2.CascadeClassifier(haar_path)
            print("OpenCV Haar Cascade loaded successfully.")
    except Exception as e:
        print(f"OpenCV Haar Cascade unavailable: {e}")
        haar_cascade = None

# =========================
# Helpers
# =========================
def allowed_image_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def image_to_base64(image: Image.Image, quality: int = 95) -> str:
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def to_grayscale_rgb(image: Image.Image) -> Image.Image:
    gray = ImageOps.exif_transpose(image).convert("L")
    gray = ImageOps.autocontrast(gray)
    return gray.convert("RGB")


def notebook_preprocess_with_preview(image: Image.Image):
    """
    Grayscale preprocessing for emotion prediction:
    - EXIF transpose
    - grayscale
    - autocontrast
    - reconvert to RGB for model compatibility
    - resize 224x224
    - float32
    - NO preprocess_input
    - NO division by 255
    """
    image = to_grayscale_rgb(image)
    model_input_pil = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    img_array = np.asarray(model_input_pil, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, model_input_pil


def make_display_preview(image: Image.Image, size=PREVIEW_SIZE, grayscale: bool = False):
    """
    Display-only preview that preserves aspect ratio.
    """
    img = to_grayscale_rgb(image) if grayscale else ImageOps.exif_transpose(image).convert("RGB")
    contained = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)
    preview = Image.new("RGB", size, PREVIEW_BACKGROUND)
    offset = ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2)
    preview.paste(contained, offset)
    return preview


def crop_square_face(rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int, extra_pad_ratio: float = FACE_CROP_PAD_RATIO):
    h, w = rgb.shape[:2]

    bw = x2 - x1
    bh = y2 - y1
    side = int(max(bw, bh) * (1.0 + extra_pad_ratio))

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(w, nx1 + side)
    ny2 = min(h, ny1 + side)

    nx1 = max(0, nx2 - side)
    ny1 = max(0, ny2 - side)

    crop = rgb[ny1:ny2, nx1:nx2]
    return crop, [int(nx1), int(ny1), int(nx2 - nx1), int(ny2 - ny1)]


def decode_base64_image(image_data: str) -> Image.Image:
    if "," not in image_data:
        raise ValueError("Invalid image format")

    _, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise ValueError("Invalid image")

    return image


def open_uploaded_image(file_storage) -> Image.Image:
    if not file_storage or file_storage.filename == "":
        raise ValueError("Empty file name")

    if not allowed_image_file(file_storage.filename):
        raise ValueError("Unsupported image type")

    image_bytes = file_storage.read()
    if not image_bytes:
        raise ValueError("Uploaded image is empty")

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise ValueError("Invalid image file")

    return image


def detect_largest_face_mediapipe(image: Image.Image):
    """
    Returns:
        face_pil, bbox, display_crop_pil
    bbox = [x1, y1, width, height]
    """
    if not MEDIAPIPE_AVAILABLE or face_detector is None:
        return None, None, None

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    results = face_detector.process(rgb)

    if not results or not results.detections:
        return None, None, None

    best_box = None
    best_area = -1

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        area = bw * bh
        if area > best_area:
            best_area = area
            best_box = (x, y, bw, bh)

    if best_box is None:
        return None, None, None

    x, y, bw, bh = best_box

    pad = int(0.12 * max(bw, bh))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    square_crop, square_bbox = crop_square_face(rgb, x1, y1, x2, y2, extra_pad_ratio=FACE_CROP_PAD_RATIO)

    if square_crop.size == 0:
        return None, None, None

    face_pil = Image.fromarray(square_crop)
    display_crop_pil = make_display_preview(face_pil, size=PREVIEW_SIZE, grayscale=True)

    return face_pil, square_bbox, display_crop_pil


def detect_largest_face_opencv(image: Image.Image):
    """
    Returns:
        face_pil, bbox, display_crop_pil
    bbox = [x1, y1, width, height]
    """
    if not CV2_AVAILABLE or haar_cascade is None:
        return None, None, None

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if faces is None or len(faces) == 0:
        return None, None, None

    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    x, y, bw, bh = faces[0]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(rgb.shape[1], x + bw)
    y2 = min(rgb.shape[0], y + bh)

    square_crop, square_bbox = crop_square_face(rgb, x1, y1, x2, y2, extra_pad_ratio=FACE_CROP_PAD_RATIO)

    if square_crop.size == 0:
        return None, None, None

    face_pil = Image.fromarray(square_crop)
    display_crop_pil = make_display_preview(face_pil, size=PREVIEW_SIZE, grayscale=True)

    return face_pil, square_bbox, display_crop_pil


def detect_face_with_fallback(image: Image.Image):
    """
    MediaPipe first, then OpenCV fallback.
    """
    face_pil, bbox, display_crop_pil = detect_largest_face_mediapipe(image)

    if face_pil is not None:
        return face_pil, bbox, display_crop_pil

    face_pil, bbox, display_crop_pil = detect_largest_face_opencv(image)
    return face_pil, bbox, display_crop_pil


def predict_emotion_from_pil(image: Image.Image, require_face: bool = True):
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_preview = image_to_base64(make_display_preview(image, size=PREVIEW_SIZE))

    face_pil, bbox, display_crop_pil = detect_face_with_fallback(image)

    # IMPORTANT: prediction only on detected face
    if face_pil is None:
        if require_face:
            raise ValueError("No face detected in the image")
        else:
            face_pil = image
            bbox = None
            display_crop_pil = make_display_preview(face_pil, size=PREVIEW_SIZE, grayscale=True)

    face_crop_preview = image_to_base64(display_crop_pil)

    processed_tensor, model_input_pil = notebook_preprocess_with_preview(face_pil)
    preprocessed_preview = image_to_base64(make_display_preview(model_input_pil, size=PREVIEW_SIZE, grayscale=True))

    preds = emotion_model.predict(processed_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    top_probability = float(preds[pred_idx])

    probabilities = [
        {
            "emotion": EMOTIONS[i],
            "probability": round(float(preds[i]) * 100, 2)
        }
        for i in range(len(preds))
    ]
    probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "predicted_class_index": pred_idx,
        "predicted_emotion": EMOTIONS[pred_idx],
        "dominant_probability": round(top_probability * 100, 2),
        "probabilities": probabilities,
        "bbox": bbox,
        "original_preview": original_preview,
        "face_crop_preview": face_crop_preview,
        "preprocessed_preview": preprocessed_preview
    }

# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict-frame", methods=["POST"])
def predict_frame():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        image = decode_base64_image(data["image"])
        result = predict_emotion_from_pil(image, require_face=True)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict-image", methods=["POST"])
def predict_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["file"]
        image = open_uploaded_image(file)

        result = predict_emotion_from_pil(image, require_face=True)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict-video", methods=["POST"])
def predict_video():
    if not CV2_AVAILABLE:
        return jsonify({"error": "OpenCV is required for video prediction"}), 500

    temp_path = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        file = request.files["file"]

        if not file or file.filename == "":
            return jsonify({"error": "Empty file name"}), 400

        if not allowed_video_file(file.filename):
            return jsonify({"error": "Unsupported video type"}), 400

        suffix = "." + file.filename.rsplit(".", 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open the video"}), 400

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return jsonify({"error": "Invalid video"}), 400

        step = max(1, frame_count // 12)
        sampled_results = []

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                try:
                    result = predict_emotion_from_pil(pil_img, require_face=True)
                    sampled_results.append(result)
                except ValueError:
                    # skip frames with no face
                    pass

                if len(sampled_results) >= 12:
                    break

            idx += 1

        cap.release()

        if not sampled_results:
            return jsonify({"error": "No face detected in the video"}), 400

        emotion_votes = {}
        prob_values = []

        for item in sampled_results:
            emotion = item["predicted_emotion"]
            emotion_votes[emotion] = emotion_votes.get(emotion, 0) + 1
            prob_values.append(float(item["dominant_probability"]))

        final_emotion = max(emotion_votes, key=emotion_votes.get)
        avg_probability = round(sum(prob_values) / len(prob_values), 2)

        return jsonify({
            "predicted_emotion": final_emotion,
            "dominant_probability": avg_probability,
            "frames_analyzed": len(sampled_results),
            "all_frame_predictions": sampled_results
        })

    except Exception as e:
        return jsonify({"error": f"Video prediction failed: {str(e)}"}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
