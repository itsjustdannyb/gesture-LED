import cv2
import os
import time

# creates folder for image class i
# the app opens up
# saves a frame every 2 seconds, it'll save a total of 100 
# it'll countdown from 1 - 10 when it's time to take shots for class i + 1
# after number of classes is complete. it'll destroy the app

def main():
    image_capture()
   


def image_capture():
    # make sure you're in the folder you want to store the images
    main_cwd = os.getcwd()
    os.chdir(main_cwd) #change directory to current working directory

    # collect user input for what to name folder
    folder_name = input("folder name: ")
    # collect number of classes from user
    num_classes = int(input("How many classes are there? "))
    image_count = int(input("How many Images per class: "))


    os.mkdir(folder_name)
    folder_path = os.path.join(main_cwd,folder_name)
    os.chdir(folder_path)

    

    for i in range(num_classes):
        class_name = input(f"Name class {i}: ") # class i in range num_class
        os.mkdir(class_name) # make class folder
        os.chdir(os.path.join(folder_path, class_name)) #change to current class directory

        print(f"Starting Class {i + 1}...")
        for countdown in range(5, 0 , -1):
            print(countdown)
            time.sleep(1)

        # start up camera
        capture = cv2.VideoCapture(0)
        last_captured = time.time()
        capture_freq = 1.5
        count = 0

        
        while True:
            ret, frame = capture.read()
            flip_frame = cv2.flip(frame, 1)

            recent_captured = time.time()
            cv2.imshow("Video", flip_frame)

            if (recent_captured - last_captured >= capture_freq and count <= image_count):
                # save image
                cv2.imwrite(f"{class_name}_{count + 1}.png", flip_frame)
                last_captured = recent_captured
                count += 1

            # to quit
            if cv2.waitKey(1) == ord("q") or count == image_count:
                break
                count = 0
                


        print(f"Stopping Class {i + 1}...")
        for countdown in range(5, 0, -1):
            print(countdown)
            time.sleep(1)

        capture.release()
        cv2.destroyAllWindows()  

        time.sleep(3)

        # go to root directory
        os.chdir(folder_path)

             

    
    
    

if __name__ == "__main__":
    main()

