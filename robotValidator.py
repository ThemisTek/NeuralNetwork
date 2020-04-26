from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar
import os
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


picar.setup()
img = cv2.VideoCapture(-1)
SCREEN_WIDTH = 160
SCREEN_HIGHT = 120
img.set(3,SCREEN_WIDTH)
img.set(4,SCREEN_HIGHT)
CENTER_X = SCREEN_WIDTH/2
CENTER_Y = SCREEN_HIGHT/2
image_size = 120


vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable = False


for layer in vgg_conv.layers:
    print(layer, layer.trainable)

model = models.Sequential()

model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))


model.load_weights("shoe_orange.h5")


val_batchsize=2
validation_dir = "./validation"

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

fnames = validation_generator.filenames

while True:
    _, bgr_image = img.read()
    orig_image = bgr_image
    bgr_image = cv2.medianBlur(bgr_image, 3)
    image_input =  np.resize(bgr_image,(1,image_size,image_size,3))
    output = model.predict(image_input)
    roundedOut = np.rint(output)
    print(roundedOut)
    cv2.imshow("Threshold lower image", bgr_image)
    k = cv2.waitKey(5) & 0xFF