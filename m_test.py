import cv2
import numpy as np
import matplotlib.pyplot as plt

def ishow(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=plt.cm.gray)
    
    
def integral(img):
    integral = img[0,:]
    for ii in range(len(img)):
        integral = integral + img[ii,:]
    integral = integral - img[0,:] 
    return integral

img = cv2.imread('./test-images/female_35.bmp',0)
ishow(img)
#img2 = cv2.imread('./test-images/male_19.bmp',0)
#ishow(img2)
from preprocessing import erosion
from preprocessing import CLAHE
from preprocessing import match_histogram as mh
from preprocessing import adaptive_threshold as at
from preprocessing import otsu
from preprocessing import laplacian
atimg = at(img,True,True,True)
ishow(atimg[0])
ishow(erosion(atimg[0],5,2,False,True))

ime = CLAHE(img)
ishow(ime)
ot = otsu(ime,1,1,1)
ishow(ot)
imgup = CLAHE(cv2.imread('Up.jpg',0),30)
ishow(imgup)

imgdown = CLAHE(cv2.imread('down.jpg',0),5)
ishow(imgdown)
ers = erosion(imgup,5,5,False,True)
ishow(ers)

integ1 = np.sum(imgup,axis = 0)
plt.plot(integ)
print("....")
integ2 = np.sum(ers,axis = 0)
plt.plot(integ2)
#f = cv2.floodFill(ers,None,(0,0),255)
#ishow(ers)
plt.plot(integ1 - integ2)

#match = mh(img,img2)
#ishow(match)

lap = (laplacian(ime,1,1))

maxi = np.max(lap)
mini = np.min(lap)
lap = np.round((lap - mini)/(maxi-mini)*255)
ishow(CLAHE(lap))
#cluster.kmeans(imgaeq,3)
#cluster.hierarchical(imgaeq,3)