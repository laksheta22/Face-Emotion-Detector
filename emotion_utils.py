import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion model
emotion_model = load_model("emotion_model.h5", compile=False)

# Emotion labels in the same order as the model's output
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Confidence threshold below which we show Neutral
CONF_THRESHOLD = 0.50

def predict_emotion(face_image):
    """
    Predict emotion from a face image.

    Parameters:
        face_image: numpy array, shape (64, 64) or (1,64,64,1)
                    Can be grayscale or BGR.

    Returns:
        emotion (str): predicted emotion label
        confidence (float): confidence of prediction (0-1)
    """

    # If image has 3 channels, convert to grayscale
    if len(face_image.shape) == 3 and face_image.shape[2] == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Resize to 64x64
    face_resized = cv2.resize(face_image, (64, 64))

    # Normalize
    face_resized = face_resized.astype("float32") / 255.0

    # Reshape to (1, 64, 64, 1) for model
    face_input = face_resized.reshape(1, 64, 64, 1)

    # Predict
    prediction = emotion_model.predict(face_input, verbose=0)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])
    emotion = emotion_labels[index]

    # Apply confidence threshold
    if confidence < CONF_THRESHOLD:
        return "Neutral", confidence

    return emotion, confidence
