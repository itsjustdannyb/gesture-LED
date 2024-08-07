from pyfirmata2 import Arduino
import time
import cv2 as cv
import mediapipe as mp
import torch
from sklearn.preprocessing import StandardScaler

print("imports ok!")

board = Arduino("COM9")
print("connected to Arduino!")

board.digital[13].write(1)
time.sleep(2)
board.digital[13].write(0)