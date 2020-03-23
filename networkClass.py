from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from cv2 import cv2
import numpy as np

class Detector :
    def __init__(self,x,y,width,height,detected,imgName):
        self.imgName = imgName
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.detected = detected

def NetworkModel():
    img_width = 45
    img_height = 45

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=( img_width, img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return model

def detector2Outut(detector):
    output = np.empty((5,1))
    output[0]=detector.x
    output[1]=detector.y
    output[2]=detector.width
    output[3]=detector.height
    output[4]=detector.detected
    return output


def detector2Input(detector):
    image = cv2.imread(detector.filename)
    image = np.resize(image,(45,45,3))
    image = np.divide(image,255)


