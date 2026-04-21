# Face Emotion Detection

Application web Flask pour la detection d'emotions faciales a partir d'une camera en direct, d'une image ou d'une video. Le projet charge un modele Keras/TensorFlow entraine et applique une detection de visage, un recadrage, un pretraitement en niveaux de gris puis une prediction parmi 7 emotions.

## Fonctionnalites

- Detection en direct depuis la webcam.
- Prediction depuis une image importee.
- Analyse d'une video par echantillonnage de plusieurs frames.
- Detection du visage avec MediaPipe si disponible, puis fallback OpenCV Haar Cascade.
- Recadrage carre du visage avant inference.
- Pretraitement coherent avec le notebook : correction EXIF, niveaux de gris, autocontrast, redimensionnement en `224x224`.
- Affichage des apercus : image originale, visage recadre et entree pretraitee.
- Retour des probabilites par classe d'emotion.

## Emotions reconnues

Par defaut, l'application utilise les classes suivantes :

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Si le fichier `model/class_indices.json` existe, il est utilise pour reconstruire la correspondance entre indices et labels.

## Structure du projet

```text
emotion-api/
|-- app.py                         # Application Flask et logique d'inference
|-- requirements.txt               # Dependances Python principales
|-- model/
|   |-- emotion_model.h5           # Modele charge par l'application
|   |-- best_emotion_model.keras   # Autre artefact de modele
|   `-- rafdb_efficientnetb2_balanced_final.keras
|-- Notebook/
|   `-- instant-emotion.ipynb      # Notebook d'entrainement/experimentation
|-- static/
|   `-- style.css                  # Styles de l'interface
`-- templates/
    `-- index.html                 # Interface web
```

## Prerequis

- Python 3.10+ recommande.
- Un modele Keras compatible disponible dans `model/emotion_model.h5`.
- Une webcam pour le mode live.
- OpenCV pour l'analyse video et le fallback de detection faciale.
- MediaPipe optionnel pour une detection de visage plus robuste.

## Installation

1. Creer et activer un environnement virtuel :

```bash
python -m venv .venv
```

Sur Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Sur macOS/Linux :

```bash
source .venv/bin/activate
```

2. Installer les dependances principales :

```bash
pip install -r requirements.txt
```

3. Installer les dependances recommandees pour la detection faciale et la video :

```bash
pip install opencv-python mediapipe
```

4. Verifier que le modele existe :

```text
model/emotion_model.h5
```

## Lancement

```bash
python app.py
```

L'application demarre par defaut sur :

```text
http://localhost:5000
```

## API

### `GET /`

Affiche l'interface web.

### `POST /predict-frame`

Recoit une image encodee en base64 depuis la webcam.

Exemple de corps JSON :

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

### `POST /predict-image`

Recoit une image dans un champ multipart appele `file`.

Formats acceptes :

```text
png, jpg, jpeg, webp, bmp
```

### `POST /predict-video`

Recoit une video dans un champ multipart appele `file`, analyse jusqu'a 12 frames contenant un visage, puis retourne l'emotion dominante.

Formats acceptes :

```text
mp4, mov, avi, mkv, webm
```

## Format de reponse

Les routes de prediction retournent notamment :

```json
{
  "predicted_class_index": 3,
  "predicted_emotion": "Happy",
  "dominant_probability": 92.45,
  "probabilities": [
    {
      "emotion": "Happy",
      "probability": 92.45
    }
  ],
  "bbox": [10, 20, 180, 180],
  "original_preview": "data:image/jpeg;base64,...",
  "face_crop_preview": "data:image/jpeg;base64,...",
  "preprocessed_preview": "data:image/jpeg;base64,..."
}
```

## Notes importantes

- La prediction est faite uniquement sur le visage detecte. Si aucun visage n'est trouve, l'API retourne une erreur.
- Le mode video necessite OpenCV.
- Si ni MediaPipe ni OpenCV ne sont disponibles, la detection faciale ne peut pas fonctionner correctement.
- Les fichiers de modeles peuvent etre volumineux. Il est recommande de ne pas les versionner directement dans Git et de les stocker dans un espace dedie si le projet est partage.

## Depannage

### `Emotion model not found`

Verifier que le fichier suivant existe :

```text
model/emotion_model.h5
```

### `No face detected`

Utiliser une image nette avec un visage visible et bien eclaire. Installer aussi les dependances recommandees :

```bash
pip install opencv-python mediapipe
```

### Erreur OpenCV en mode video

Installer OpenCV :

```bash
pip install opencv-python
```

## Licence

Ajouter ici la licence choisie pour le projet avant publication.
