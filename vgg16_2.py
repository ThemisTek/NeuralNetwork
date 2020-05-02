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

print(device_lib.list_local_devices())

image_size = 224


#input_shape=(image_size, image_size, 3)
model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

model_vgg16_conv.summary()

model = Sequential()
for l in model_vgg16_conv.layers[0:-1]:
      model.add(l)

for l in model.layers:
      l.trainable = False

model.add(Dense(3,activation="softmax"))

model.summary()

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "./train"
validation_dir = "./validation"
train_batchsize = 100

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

val_batchsize=100

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

fnames = validation_generator.filenames

label2index = validation_generator.class_indices

idx2label = dict((v,k) for k,v in label2index.items())

print(idx2label)
label2indexTrain = train_generator.class_indices

print(label2indexTrain)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

# train_generator.samples/train_generator.batch_size

history = model.fit()


model.save('bananaModel.h5')
model.save_weights(filepath='bananaWeights.h5')