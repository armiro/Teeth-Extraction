#Initialization
import numpy as np
import cv2
import matplotlib.pyplot as plt


#Load Image
img = cv2.imread('5.jpg',0)

#Gradient
kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.figure(figsize=(7, 7))
plt.imshow(gradient, cmap=plt.cm.gray)


thl = cv2.adaptiveThreshold(gradient,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,13)    
plt.figure(figsize=(17, 17))
plt.imshow(thl, cmap=plt.cm.gray)

 
    
#TopHat

blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(30,30))
blackhis = cv2.equalizeHist(blackhat)
plt.figure(figsize=(7, 7))
plt.imshow(blackhis, cmap=plt.cm.gray)


