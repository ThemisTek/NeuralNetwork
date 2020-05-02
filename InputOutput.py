import numpy as np
import os
from cv2 import cv2
from glob import glob
from keras.applications.vgg16 import preprocess_input


def convertImageLinkToArray(loc,dim):
    im = cv2.imread(loc)
    
    imR = np.array(cv2.resize(im,(dim,dim)),"float")
    imR = preprocess_input(imR.astype('float32'))
    return imR


def validateExtension(ex):
    ext = [".jpg" , ".png"] 
    for e in ext:
        if(ex.endswith(e)):
            return True
    return False


class InputGen:
    def __init__ (self,folder = "./train",dimension = 224) :
        ext = [".jpg" , ".png"]
        subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
        names = []
        for s in subfolders:
            names.append(s.split("\\")[1])
        totalLen = 0
        tempDictionary = {}
        for n in names :
            tempDictionary[n] = []
            subFolderPath = folder + "/" + n
            filesInFolder = os.listdir(subFolderPath)
            for imName in filesInFolder:
                for e in ext:
                    if(imName.endswith(e)):
                        totalLen +=1
                        tempDictionary[n].append(subFolderPath + "/" + imName)

        dicKeys = list(tempDictionary.keys())
        inpt = np.empty((totalLen,dimension,dimension,3))
        outp = np.zeros((totalLen,len(dicKeys)))
        outputInd = 0
        arrayIndx = 0

        for k in dicKeys:
            for im in tempDictionary[k]:
                if(validateExtension(im)):
                    convertedIm = convertImageLinkToArray(im,dimension)
                    inpt[arrayIndx,:] = convertedIm
                    outp[arrayIndx,outputInd] = 1
                    arrayIndx +=1
            outputInd +=1
        self.input = inpt
        self.output = outp


