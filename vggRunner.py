from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
from cv2 import cv2
import numpy as np

image_size = 224

vgg_conv = VGG16(weights='imagenet', include_top=True, input_shape=(image_size, image_size, 3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable = False


for layer in vgg_conv.layers:
    print(layer, layer.trainable)

model = models.Sequential()

model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

dirpath = os.getcwd()

model.load_weights("shoe_orange.h5")

folderPath = dirpath + r"\Testing\*.jpg"

print(folderPath)

filenames = [img for img in glob.glob(folderPath)]

print (filenames)

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


label2index = validation_generator.class_indices
idx2label = dict((v,k) for k,v in label2index.items())
print(idx2label)

images = []
for img in filenames:
    im = cv2.imread(img)
    resized = np.resize(im,(image_size,image_size,3))
    reshaped = np.reshape(resized,(1,image_size,image_size,3))
    print(reshaped.shape)
    result = model.predict(reshaped)
    cv2.imshow(img,im)
    print(img,result)

