import networkClass
import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from networkClass import Detector
import tensorflow as tf

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(type(train_labels),train_labels[0])

print(train_images.shape,train_labels.shape)

np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

objectReader = np.load("test.npy")

model = networkClass.NetworkModel()

d = networkClass.dimensions()
width = d[0]
height = d[1]
inputSize = len(objectReader)
inputs=np.empty((inputSize,width,height,3))
outputs = np.empty((inputSize,5,1))


i=0

for o in objectReader:
    outputs[i] = networkClass.detector2Output(o)
    inputs[i] = networkClass.detector2Input(o)
    i=i+1


print(outputs.shape)
print(inputs.shape)


labels = tf.reshape(outputs, [inputSize,5])

print("printing layers")

for layer in model.layers:
    print(layer.input_shape, layer.output_shape)

model.fit(inputs, labels, steps_per_epoch=10000,epochs = 1)
predictions = model.predict(inputs)
print(predictions)
print(outputs)


model.save("test_model.h5")



