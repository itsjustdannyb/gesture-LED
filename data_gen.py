import os
import cv2
import mediapipe as mp 
import pickle
import numpy as np

image_folder_path = r"C:\Users\Daniel\Documents\hardware projects\gesture-LED\hands_data"

# path = os.getcwd()
# folder = "images_with_landmarks"
# os.mkdir(folder)
# lm_folder = os.path.join(path, folder)
# os.chdir(lm_folder)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:

    data = []
    labels = []

    for class_name in os.listdir(image_folder_path):
        img_path = os.path.join(image_folder_path, class_name)

        data_tmp = []

        for img in os.listdir(img_path):
            img_file_path = os.path.join(img_path, img)
            image = cv2.imread(img_file_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_tmp.append(x)
                        data_tmp.append(y)

                data.append(data_tmp)
                labels.append(class_name)



f = open("training_data.pickle", "wb")
pickle.dump({"data":data, "labels":labels}, f)
f.close()

# data = np.array(data)
# labels = np.array(labels)

# p = np.random.permutation(len(data))
# data = data[p]
# labels = labels[p]

# np.savez("training_data.npz", land_marks=data, labels=labels)


            # cv2.imwrite(f"{count}.png", image)
            # count += 1

        # os.chdir(lm_folder)

print("all landmarks added successfully!")