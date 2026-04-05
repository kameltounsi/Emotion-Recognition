# Face Emotion Detection

A complete AI-powered facial emotion detection web application built with **Flask**, **TensorFlow/Keras**, **MediaPipe**, and **OpenCV**.

This project supports:

- **Live webcam emotion detection**
- **Image upload emotion prediction**
- **Video upload emotion prediction**
- **Face detection and face cropping**
- **Improved face preview**
- **Preprocessed image visualization**
- **Emotion probability display**

---

## Features

### 1. Live Camera Detection
- Real-time webcam emotion detection
- Face detection with bounding box
- Predicted dominant emotion
- Dominant probability display
- Preview of:
  - original frame
  - detected face crop
  - preprocessed image used by the model

### 2. Image Prediction
- Upload an image
- Detect and crop the face
- Apply the same preprocessing used in training
- Predict facial emotion
- Show probabilities for all emotion classes

### 3. Video Prediction
- Upload a video
- Sample frames from the video
- Detect and crop faces from selected frames
- Predict emotions frame by frame
- Return final dominant emotion based on voting

---

## Model

This project uses a trained deep learning model saved as:

```bash
model/emotion_model.h5
