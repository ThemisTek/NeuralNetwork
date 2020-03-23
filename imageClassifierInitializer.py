import numpy as np
from cv2 import cv2
import glob
import os
 
dirpath = os.getcwd()
print(dirpath)

class Detector :
    def __init__(self,x,y,width,height,detected,imgName):
        self.imgName = imgName
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.detected = detected

folderPath = dirpath + r"\Capture\*.png"

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
refPt =[(0,0),(0,0)]
cropping = False

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP :
        refPt.append((x, y))
        cropping = False
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

i=0
imageName=filenames[i]
image = cv2.imread(imageName)

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
		
while True:
    cv2.imshow("image",image)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord("c")):
        break
    elif(key == ord("n")):
        x1=refPt[0][0]
        y1=refPt[0][1]
        x2=refPt[1][0]
        y2=refPt[1][1]
        print(refPt)
        dimensions = image.shape
        x = dimensions[0]
        y = dimensions[1]
        width = (x2-x1)/x
        height = (y2-y1)/y
        posX = x1/x + width/2
        posY = y1/y + height/2       
        detected = not(refPt[0][0] == 0 and refPt[0][1]==0)
        refPt =[(0,0),(0,0)]
        dec = Detector(posX,posY,width,height,detected,imageName)
        imagesObject.append(dec)
        print(posX,posY,width,height,detected)
        i=i+1
        if(i >= len(filenames)):
            break
        else:
            imageName = filenames[i]
            image = cv2.imread(imageName)







np.save("test.npy",imagesObject)
