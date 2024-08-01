import pickle
import numpy as np

import cv2
import mediapipe as mp

import torch
import torch.nn as nn
import torch.nn.functional as F

# hand detection model
from model_arch import Net

from sklearn.preprocessing import StandardScaler
print("imports ok!")

# for data preprocessing
with open("scaler.pickle", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
# scaler = pickle.load("scaler.pickle")

# the hand classifier
model = Net()
state_dict = torch.load("hand-classifier") # load model's state dictionary
model.load_state_dict(state_dict)
model.eval()

# for capturing data and drawing landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# get input for camera
capture = cv2.VideoCapture(0)

# data = []

while True:
    ret, frame = capture.read()
    # cv2.imshow("Video", frame)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            
            
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data.append(x)
                data.append(y)
                # data.append(data_tmp)

            # reshape data
            data = np.array(data).reshape(1, -1)

            data = scaler.transform(data)
            # data = data.reshape(1, -1)
            data = np.asarray(data, dtype="float32")
            data = torch.from_numpy(data).squeeze(0)

            with torch.no_grad():
                prediction = model(data)
                if ((prediction < 0.5).float() == 0):
                    print("Hands are Open!")
                else:
                    print("Hands are Closed!")


    if cv2.waitKey(1) == ord("q"):
        break



capture.release()
cv2.destroyAllWindows()