import os
import mediapipe as mp
import cv2 
import pickle

image_folder_path = r"C:\Users\Daniel\Documents\hardware projects\gesture-LED\hands_data"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

for dir_ in os.listdir(image_folder_path):
    for img_path in os.listdir(os.path.join(image_folder_path, dir_)):
        data_tmp = []
        img = cv2.imread(os.path.join(image_folder_path, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_tmp.append(x)
                    data_tmp.append(y)
            data.append(data_tmp)
            labels.append(dir_)
            

f = open("dataset.pickle", "wb")
pickle.dump({"data":data, "labels":labels}, f)
f.close()


