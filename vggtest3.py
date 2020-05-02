from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
from tensorflow.python.client import device_lib
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense,Dropout
from keras.models import Model,Sequential
from keras.optimizers import Adam
import keras
import numpy as np
import InputOutput

print(device_lib.list_local_devices())

image_size = 224




model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

model_vgg16_conv.summary()

model = Sequential()
for l in model_vgg16_conv.layers[0:-1]:
      model.add(l)

for l in model.layers:
      l.trainable = False

model.add(Dense(3,activation="softmax"))

model.summary()


train = InputOutput.InputGen()
validation = InputOutput.InputGen(folder ="./validation")

xTrain = train.input
yTrain = train.output

xVal = validation.input
yVal = validation.output

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.fit(xTrain,yTrain,epochs = 20)


model.save('custom.h5')
model.save_weights(filepath='customWeights.h5')
