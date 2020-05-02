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

model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

model_vgg16_conv.summary()

model = Sequential()
for l in model_vgg16_conv.layers[0:-1]:
      model.add(l)

model.add(Dense(3,activation="softmax"))
model.summary()

model.load_weights("customWeights.h5")

train = InputOutput.InputGen()
validation = InputOutput.InputGen(folder ="./validation")

xTrain = train.input
yTrain = train.output

xVal = validation.input
yVal = validation.output

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])


lxVal = len(xVal)

prediction = model.predict(xVal)
correct 


for i in range(lxVal):
    predicted = np.argmax(prediction[i])
    correct = yVal[i]
    correctPrediction = np.argmax(correct)
    print(correctPrediction,predicted,prediction[i][predicted])

