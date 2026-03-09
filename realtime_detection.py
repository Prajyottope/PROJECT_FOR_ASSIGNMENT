import cv2
import joblib
import time
from collections import Counter

from utils.feature_extraction import extract_features

model = joblib.load("models/emotion_model.pkl")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cam = cv2.VideoCapture(0)

predictions = []
start_time = time.time()

final_emotion = "Analyzing..."

while True:

    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face)

        for (ex,ey,ew,eh) in eyes:

            eye = face[ey:ey+eh, ex:ex+ew]

            features = extract_features(eye)

            prediction = model.predict([features])[0]

            predictions.append(prediction)

            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

    elapsed = time.time() - start_time

    if elapsed >= 5 and len(predictions) > 0:

        most_common = Counter(predictions).most_common(1)[0][0]

        final_emotion = most_common

        predictions = []

        start_time = time.time()

    cv2.putText(
        frame,
        f"Emotion: {final_emotion}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Eye Emotion Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()