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

import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

# for data preprocessing
with open("scaler.pickle", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
# scaler = pickle.load("scaler.pickle")

# the hand classifier
model = Net()
state_dict = torch.load("hand-classifier") # load model's state dictionary
model.load_state_dict(state_dict)
model.eval()

# get input for camera
capture = cv2.VideoCapture(0)

# for capturing data and drawing landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)


classes = {0:"Open", 1:"Closed"}
prediction = ""

while True:
    # data = []

    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)


    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # cv2.imshow("video", frame)

        for hand_landmarks in results.multi_hand_landmarks:
            data = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data.append(x)
                data.append(y)
                # data.append(data_tmp)

            # reshape data
            # data = np.array(data).reshape(1, -1)

            data = scaler.transform([data])
            data = np.asarray(data, dtype="float32")
            data = torch.from_numpy(data).squeeze(0)

            with torch.no_grad():
                prediction = model(data)
                prediction = prediction.numpy() # CONVERT TENSOR TO NUMPY
                # print(prediction)
                prediction = classes[(prediction[0] > 0.5)]
                print(prediction)

    cv2.putText(frame, prediction, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv2.imshow("Video",frame)
    # cv2.waitKey(25)


    if cv2.waitKey(1) == ord("q"):
        break



capture.release()
cv2.destroyAllWindows()



