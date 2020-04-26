import numpy as np
from cv2 import cv2
import glob
import os
from networkClass import CircleCrossDetector
 
dirpath = os.getcwd()
print(dirpath)

folderPath = dirpath + r"\Capture\Circle\*.png"

print(folderPath)


filenames = [img for img in glob.glob(folderPath)]

print (filenames)

images = []
for img in filenames:
    n= cv2.imread(img)
    images.append(n)
    print (img)


imagesObject = []
image = []


i=0
imageName=filenames[i]
image = cv2.imread(imageName)

cv2.namedWindow("image")



		
while True:
    cv2.imshow("image",image)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord("c")):
        break
    elif(key == ord("b")):
        dec = CircleCrossDetector(1,0,0,imageName)
        imagesObject.append(dec)
        i=i+1
        if(i >= len(filenames)):
            break
        else:
            imageName = filenames[i]
            image = cv2.imread(imageName)

    elif(key == ord("n")):
        dec = CircleCrossDetector(0,1,0,imageName)
        imagesObject.append(dec)
        i=i+1
        if(i >= len(filenames)):
            break
        else:
            imageName = filenames[i]
            image = cv2.imread(imageName)

    elif(key == ord("m")):
        dec = CircleCrossDetector(0,0,1,imageName)
        imagesObject.append(dec)
        i=i+1
        if(i >= len(filenames)):
            break
        else:
            imageName = filenames[i]
            image = cv2.imread(imageName)




np.save("CircleRead.npy",imagesObject)
