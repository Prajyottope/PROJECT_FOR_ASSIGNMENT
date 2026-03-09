import cv2
import os

emotion = input("Enter emotion name (happy/sad/angry/surprise/neutral): ")

save_path = f"dataset/{emotion}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cam = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face)

        for (ex,ey,ew,eh) in eyes:

            eye = face[ey:ey+eh, ex:ex+ew]

            file_name = f"{save_path}/{count}.jpg"

            cv2.imwrite(file_name, eye)

            count += 1

            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

    cv2.imshow("Collecting Eye Dataset", frame)

    if cv2.waitKey(1) == 27 or count > 20:
        break

cam.release()
cv2.destroyAllWindows()