from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar
import os
import networkClass

picar.setup()
img = cv2.VideoCapture(-1)
SCREEN_WIDTH = 160
SCREEN_HIGHT = 120
img.set(3,SCREEN_WIDTH)
img.set(4,SCREEN_HIGHT)
CENTER_X = SCREEN_WIDTH/2
CENTER_Y = SCREEN_HIGHT/2

model = networkClass.NetworkModel()
model.load_weights("test_model.h5")

while True:
    _, bgr_image = img.read()
    orig_image = bgr_image
    bgr_image = cv2.medianBlur(bgr_image, 3)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    cv2.imshow("Threshold lower image", bgr_image)
    k = cv2.waitKey(5) & 0xFF