import cv2
import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from ROI_revision import *
from ROI_extraction import *

img_address = './test-images/2.bmp'
img = cv2.imread(img_address, 0)

initial_roi = extract_roi(image=img, return_result=1)
final_roi = revise_boundaries(image=initial_roi, return_result=1)

plt.subplot(2, 1, 1), plt.imshow(X=img, cmap='gray')
plt.subplot(2, 1, 2), plt.imshow(X=final_roi, cmap='gray')
plt.show()
