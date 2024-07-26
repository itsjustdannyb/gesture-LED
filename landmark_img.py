import os
import cv2
import mediapipe as mp 

image_folder_path = r"C:\Users\Daniel\Documents\hardware projects\gesture-LED\hands_data"

path = os.getcwd()
folder = "images_with_landmarks"
os.mkdir(folder)
lm_folder = os.path.join(path, folder)
os.chdir(lm_folder)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:

    for class_name in os.listdir(image_folder_path):
        img_path = os.path.join(image_folder_path, class_name)
        os.mkdir(f"{class_name}")
        class_dir = os.path.join(lm_folder, class_name)
        os.chdir(class_dir)
        print(f"Now saving to >>>>>> {os.getcwd()}")
        
        count = 0


        for img in os.listdir(img_path):
            img_file_path = os.path.join(img_path, img)
            image = cv2.imread(img_file_path)


            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand in (results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)


            cv2.imwrite(f"{count}.png", image)
            count += 1

        os.chdir(lm_folder)

print("all landmarks added successfully!")