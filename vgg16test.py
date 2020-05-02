from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
from tensorflow.python.client import device_lib
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input, Flatten, Dense

print(device_lib.list_local_devices())

image_size = 224

#input_shape=(image_size, image_size, 3)
vgg_conv = VGG16(weights='imagenet', include_top=False, )

for layer in vgg_conv.layers:
    layer.trainable = False


for layer in vgg_conv.layers:
    print(layer, layer.trainable)

x = Flatten(name='flatten')(vgg_conv.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(3, activation='softmax', name='predictions')(x)

input = Input(shape=(3,image_size,image_size),name = 'image_input')

my_model = Model(input=input, output=x)

my_model.summary()


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
train_batchsize = 80

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

val_batchsize=20

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

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)


model.save('small_last4.h5')
model.save_weights(filepath='shoe_orange.h5')