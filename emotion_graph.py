import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from collections import deque

# -----------------------------
# Load Model
# -----------------------------
model = load_model(r"C:\Emotion_Robot\model.hdf5")  # Update path to your model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -----------------------------
# Initialize Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------------
# Matplotlib setup
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
max_history = 50  # number of frames to show
history = {label: deque([0]*max_history, maxlen=max_history) for label in emotion_labels}

lines = {}
for label in emotion_labels:
    line, = ax.plot(history[label], label=label)
    lines[label] = line

ax.set_ylim([0,1])
ax.set_xlim([0, max_history])
ax.set_ylabel("Probability")
ax.set_xlabel("Frame History")
ax.set_title("Emotion Probabilities Over Time")
ax.legend(loc='upper right')

print("[INFO] Starting webcam. Press 'q' to quit.")

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(preds)]

        # Draw rectangle and label
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Update history and line graph
        for i, lbl in enumerate(emotion_labels):
            history[lbl].append(preds[i])
            lines[lbl].set_ydata(history[lbl])
        fig.canvas.draw()
        fig.canvas.flush_events()

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
print("[INFO] Webcam closed.")
