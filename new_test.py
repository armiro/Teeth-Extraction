import cv2
import preprocessing
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('./test-auto-cropped/2.bmp', 0)
img = preprocessing.imfill(image=img, threshold=10, return_result=1)
img = np.array(img, dtype='uint8')
img = cv2.blur(img, (50, 50))

plt.imshow(img, cmap='gray')
plt.show()
plt.imsave(fname='C://Users/arman/Desktop/tmp.jpg', arr=img, cmap='gray')
