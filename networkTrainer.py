import networkClass
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from networkClass import Detector

np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

objectReader = np.load("test.npy")

model = networkClass.NetworkModel()

for o in objectReader:
    print(o.imgName)



