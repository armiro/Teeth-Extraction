import cv2
import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from ROI_revision import *
from ROI_extraction import *

img_address = './test-images/2.bmp'
img = cv2.imread(img_address, 0)
print('original image dimensions:', img.shape)
initial_roi, initial_boundaries = extract_roi(image=img, return_result=1)
print('initial ROI dimensions:', initial_roi.shape)
revised_roi, revised_boundaries = revise_boundaries(image=initial_roi, return_result=1)
print('final ROI dimensions:', revised_roi.shape)

# print(initial_boundaries)
# print(final_boundaries)

upper_height = initial_boundaries[3] + revised_boundaries[3]
left_width = initial_boundaries[0] + revised_boundaries[0]
lower_height = img.shape[0] - initial_boundaries[2] + revised_boundaries[2]
right_width = img.shape[1] - initial_boundaries[1] + revised_boundaries[1]

up_left_corner = (upper_height, left_width)
up_right_corner = (upper_height, right_width)
down_left_corner = (lower_height, left_width)
down_right_corner = (lower_height, right_width)

print('roi points:', up_left_corner, up_right_corner, down_left_corner, down_right_corner)

plt.subplot(2, 1, 1), plt.imshow(X=img, cmap='gray')
plt.subplot(2, 1, 2), plt.imshow(X=revised_roi, cmap='gray')
plt.show()
