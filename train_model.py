import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from utils.feature_extraction import extract_features

dataset_path = "dataset"

X = []
y = []

emotions = os.listdir(dataset_path)

for emotion in emotions:

    emotion_path = os.path.join(dataset_path, emotion)

    for img_name in os.listdir(emotion_path):

        img_path = os.path.join(emotion_path, img_name)

        img = cv2.imread(img_path, 0)

        if img is None:
            continue

        features = extract_features(img)

        X.append(features)
        y.append(emotion)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = SVC(kernel="linear", probability=True)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print(classification_report(y_test,pred))

joblib.dump(model,"models/emotion_model.pkl")

print("Model saved successfully.")