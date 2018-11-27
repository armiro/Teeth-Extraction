#Initialization
import numpy as np
import cv2
import matplotlib.pyplot as plt


#Load Image
img = cv2.imread('2.jpg',0)

showeq = 1
showth = 1
showoust = 0
#histogrm calc
hist = cv2.calcHist(img,[0],None,[256],[0,256])



#histogrm equlization
imgeq = cv2.equalizeHist(img)
hist = cv2.calcHist(imgeq,[0],None,[256],[0,256])



#Adaptive Histogram Equlization
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(25,25))
imgaeq = clahe.apply(img)
hist = cv2.calcHist(imgaeq,[0],None,[256],[0,256])


if showeq==1:
    plt.hist(img.ravel(),256,[0,256]); 
    plt.show()
    plt.hist(imgeq.ravel(),256,[0,256]); 
    plt.show()
    plt.hist(imgaeq.ravel(),256,[0,256]); 
    plt.show()
    plt.figure(figsize=(7, 7))
    plt.imshow(img, cmap=plt.cm.gray)
    plt.figure(figsize=(7, 7))
    plt.imshow(imgeq, cmap=plt.cm.gray)
    plt.figure(figsize=(7, 7))
    plt.imshow(imgaeq, cmap=plt.cm.gray)
    
    
#Adaptive Thresholding
ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_TRUNC)
ret,thg = cv2.threshold(th1,50,255,cv2.THRESH_TOZERO)
thl = cv2.adaptiveThreshold(imgaeq,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,17)    

#Remove noise
thln = cv2.GaussianBlur(thl,(5,5),0)
thln2 = cv2.adaptiveThreshold(thln,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,17)    

if showth==1:
    plt.figure(figsize=(7, 7))
    plt.imshow(thg, cmap=plt.cm.gray)
    plt.figure(figsize=(7, 7))
    plt.imshow(thl, cmap=plt.cm.gray)
    plt.figure(figsize=(7, 7))
    plt.imshow(thln, cmap=plt.cm.gray)


#Ousto Thresholding
blur = cv2.GaussianBlur(imgaeq,(5,5),0)
ret3,thoust = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
if showoust==1:
    plt.figure(figsize=(7, 7))
    plt.imshow(thoust, cmap=plt.cm.gray)
    
    
    
    
#canny edge detection
imgaeqln=cv2.GaussianBlur(imgaeq,(5,5),0)
edges = cv2.Canny(imgaeqln,140,200)
plt.figure(figsize=(7, 7))
plt.imshow(edges, cmap=plt.cm.gray)

#LAplacian


laplacian = cv2.Laplacian(imgaeq,cv2.CV_64F)
plt.figure(figsize=(7, 7))
plt.imshow(laplacian, cmap=plt.cm.gray)