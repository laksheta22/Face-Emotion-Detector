import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# --------------------------
# Load pre-trained model
# --------------------------
emotion_model = load_model("emotion_model.h5", compile=False)

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --------------------------
# Settings
# --------------------------
face_queues = {}  # smoothing per face
emotion_colors = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 128),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 0),
    "Sad": (255, 0, 0),
    "Surprise": (0, 255, 255),
    "Neutral": (200, 200, 200)
}

# For graph visualization (last 50 frames per emotion)
graph_history = {e: deque(maxlen=50) for e in emotion_labels}

# --------------------------
# Helper functions
# --------------------------
def predict_emotion(face_image):
    """face_image shape: (1, 64, 64, 1)"""
    face_image = face_image.astype("float32") / 255.0
    prediction = emotion_model.predict(face_image, verbose=0)
    index = np.argmax(prediction)
    emotion = emotion_labels[index]
    confidence = float(prediction[0][index])
    return emotion, confidence

def draw_label(frame, text, x, y, color):
    """Draw rectangle with text above face"""
    cv2.rectangle(frame, (x, y - 30), (x + 200, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_graph(frame, history):
    """
    Draw a horizontal bar chart for emotion history.
    Each emotion has its own row, bars grow upwards.
    """
    start_x = frame.shape[1] - 300  # space from right
    start_y = 50                     # top margin
    bar_width = 5                     # width of each small bar
    height_scale = 100                # height multiplier
    spacing_y = 40                    # space between emotion rows

    for i, (emotion, values) in enumerate(history.items()):
        row_y = start_y + i * spacing_y
        for j, val in enumerate(values):
            bar_height = int(val * height_scale)
            cv2.rectangle(frame,
                          (start_x + j*bar_width, row_y),
                          (start_x + j*bar_width + bar_width, row_y - bar_height),
                          emotion_colors[emotion], -1)
        # Draw emotion label at the start of each row
        cv2.putText(frame, emotion, (start_x - 80, row_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# --------------------------
# Main loop
# --------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for idx, (x, y, w, h) in enumerate(faces):
        face = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        face_input = face_resized.reshape(1, 64, 64, 1)

        # Predict emotion + confidence
        emotion, confidence = predict_emotion(face_input)

        # Smooth per face
        if idx not in face_queues:
            face_queues[idx] = deque(maxlen=7)
        face_queues[idx].append(emotion)
        final_emotion = max(set(face_queues[idx]), key=face_queues[idx].count)

        # Bounding box
        color = emotion_colors.get(final_emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        # Label with confidence
        label = f"{final_emotion} {confidence*100:.1f}%"
        draw_label(frame, label, x, y, color)

        # Update graph history
        for e in graph_history.keys():
            graph_history[e].append(confidence if e == final_emotion else 0)

    # Draw graph
    draw_graph(frame, graph_history)

    cv2.imshow("Emotion Detector - Press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
