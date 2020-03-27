from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from cv2 import cv2
import numpy as np
import keras

def dimensions():
    return [45,45]

class Detector :
    def __init__(self,x,y,width,height,detected,imgName):
        self.imgName = imgName
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.detected = detected

def NetworkModel():
    d = dimensions()
    img_width = d[0]
    img_height = d[1]

    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), input_shape=( img_width, img_height,3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(5))
    # model.add(Activation('sigmoid'))
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(img_width, img_height,3)),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(units = 5)
    # ])

    model = keras.Sequential()
    model.add(Conv2D(64,kernel_size=1,input_shape=(img_width,img_height,3)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(units=5))


    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def detector2Output(detector):
    output = np.empty((5,1))
    output[0,0]=detector.x
    output[1,0]=detector.y
    output[2,0]=detector.width
    output[3,0]=detector.height
    output[4,0]=detector.detected
    return output


def detector2Input(detector):
    image = cv2.imread(detector.imgName)
    image = np.resize(image,(45,45,3))
    image = np.divide(image,255)
    return image


