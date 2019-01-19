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
from preprocessing import niblack_and_sauvola
# CLAHE
img = CLAHE(img)
ishow(img)

#Binary

nb,sa = niblack_and_sauvola(img,91,0,1)

ishow(nb)
ishow(sa)

black_classes = np.zeros([len(sa),len(sa[0])])
for ii in range(len(sa)):
    for jj in range(len(sa[0])):
        if sa[ii,jj] == 0:
            print(ii,jj)
            first_black = [ii,jj]
            break
    if sa[ii,jj] == 0:
        break   
black_group = [first_black]

center = [1,16]
for ii in range(3):
    ii = ii-1
    for jj in range(3):
        jj = jj-1        
        if sa[center[0]+ii,center[1]+jj] == False:
           black_group.append([center[0]+ii,center[1]+jj]) 

k=black_group
k.sort()
black_group = list(k for k,_ in itertools.groupby(k))
     
for pixel in black_group:
    sa[pixel] = True
ishow(sa)