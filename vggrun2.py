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
from keras.layers import Input, Flatten, Dense
from keras.models import Model ,Sequential
from keras.models import load_model
import numpy as np
from cv2 import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import keras

image_size = 224

model_vgg16_conv = keras.applications.vgg16.VGG16()

model_vgg16_conv.summary()

model = Sequential()
for l in model_vgg16_conv.layers[0:-1]:
      model.add(l)

for l in model.layers:
      l.trainable = False

model.add(Dense(3,activation="softmax"))

model.load_weights("bananaWeights.h5")

dirpath = os.getcwd()


folderPath = dirpath + r"\Testing\*.jpg"

print(folderPath)

filenames = [img for img in glob.glob(folderPath)]

print (filenames)

val_batchsize=50
validation_dir = "./validation"

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Create a generator for prediction
# validation_generator = validation_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(image_size, image_size),
#         batch_size=val_batchsize,
#         class_mode='categorical',
#         shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
    
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])
    

    print(title)


