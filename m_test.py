import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

def ishow(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=plt.cm.gray)
    
    
def integral(img):
    integral = img[0,:]
    for ii in range(len(img)):
        integral = integral + img[ii,:]
    integral = integral - img[0,:] 
    return integral

img = cv2.imread('./test-images/female_35_crop.jpg',0)
ishow(img)
#img2 = cv2.imread('./test-images/male_19.bmp',0)
#ishow(img2)
from preprocessing import erosion
from preprocessing import CLAHE
from preprocessing import match_histogram as mh
from preprocessing import adaptive_threshold as at
from preprocessing import otsu
from preprocessing import laplacian
from preprocessing import niblack
from preprocessing import sauvola



# CLAHE
img = CLAHE(img)
#ishow(img)

#Binary

sa = sauvola(img,91,0,1)
sa = np.invert(sa)
sa = np.array(sa,dtype = 'uint8')
nb = niblack(img,91,1,0,1)
#ishow(nb)
#ishow(sa)

#Labels
retval, labels = cv2.connectedComponents(sa)  
   
sa = sauvola(img,91,0,1)
sa = np.array(sa,dtype = 'uint8')

hist=np.histogram(labels,retval)[0]
noise= list()
th = 20
hist_copy = np.copy(hist)
hist_copy = sorted(hist_copy,reverse = True)
th = hist_copy [5]

for ii in range(len(hist)):
    if hist[ii]>th:
        noise.append(ii)
gaps = np.ones([len(sa),len(sa[0])],np.bool)
#for ii in range(len(sa)):
#    for jj in range((len(sa[0]))):
#        if labels[ii,jj] != 0:
#            for kk in range(len(hist)):
#                if labels[ii,jj] == hist[kk]:
#                    gaps[ii,jj] = True

noise = noise[1:]
for ii in range(len(sa)): 
    print("---------",ii)
    for num in noise:
        indexes = [i for i, j in enumerate(labels[ii]) if j == num] 
        for jj in indexes:
            gaps[ii,jj] = False
            print(jj)

            
ishow(gaps)           
    
               
              


