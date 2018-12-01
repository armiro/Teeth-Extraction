import cv2
import numpy as np
import matplotlib.pyplot as plt

def ishow(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=plt.cm.gray)

img = cv2.imread('./test-images/photo_2018-11-01_22-42-46.jpg',0)
ishow(img)


from preprocessing import erosion
from preprocessing import CLAHE


ime = CLAHE(img)
ishow(ime)

ers = erosion(ime,5,1,False,True)
ishow(ers)

f = cv2.floodFill(ers,None,(0,0),255)
ishow(ers)

