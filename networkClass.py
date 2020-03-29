from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from cv2 import cv2
import numpy as np
import keras

def dimensions(mode = 1):
    return [45,45]

class Detector :
    def __init__(self,x,y,width,height,detected,imgName):
        self.imgName = imgName
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.detected = detected

class BlackWhiteDetector :
    def __init__(self,x,y,up,down,notDetected,imgName):
        self.imgName = imgName
        self.up = up
        self.down = down
        self.notDetected = notDetected


def NetworkModel(mode = 1):
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

    model = []

    if(mode == 1):
        model = keras.Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=(img_width,img_height,3)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
    elif(mode == 2):
        model = keras.Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=(img_width,img_height,1)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])


    



    

    return model

def detector2Output(detector,mode = 1):
    output = np.empty((5,1))
    if(mode == 1):
        output[0,0]=detector.x
        output[1,0]=detector.y
        output[2,0]=detector.width
        output[3,0]=detector.height
        output[4,0]=detector.detected
    elif(mode == 2):
        output = np.empty((3,1))
        output[0,0] = detector.up
        output[0,1] = detector.down
        output[0,2] = detector.notDetected
    
    return output


def detector2Input(detector,mode = 1):
    image = cv2.imread(detector.imgName)
    d = dimensions()
    img_width = d[0]
    img_height = d[1]
    if(mode == 1):
        image = np.resize(image,(img_width,img_height,3))
        image = np.divide(image,255)
    elif(mode == 2):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = np.resize(image,(img_width,img_height,1))
        image = image.astype(float)
        image = np.divide(image,255)

    return image


