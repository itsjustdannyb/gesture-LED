import mediapipe as mp 
import cv2 as cv2

mp_drawing = mp.solutions.drawing_utils # to render landmarks
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence=0.5, max_num_hands=4) as hands:

        while True:

            ret, frame = cap.read()

            flip_img = cv2.flip(frame, 1)

            # convert image to RGB
            image = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # to avoid altering

            results = hands.process(image)

            # convert image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(flip_img, cv2.COLOR_RGB2BGR)

            # render results
            if results.multi_hand_landmarks:
                for hand in (results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(flip_img, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(flip_img, "bams", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 250), 4)            
            cv2.imshow("Video", flip_img)

            if  cv2.waitKey(1) == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()