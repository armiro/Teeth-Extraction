import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

from preprocessing import quadtree_decomp, CLAHE, match_histogram

img = cv2.imread('./test-images/female_28.bmp', 0)
source = copy.deepcopy(img)
source_height, source_width = source.shape[0], source.shape[1]
middle_width = int(source_width/2.)

left_shadow = source[:, int(5*middle_width/6.):int(29*middle_width/30.)]
template = source[:, :int(source_width/3.)]
matched = match_histogram(source=left_shadow, template=template)
# source = np.hstack((one_third, two_third, last_third))

plt.subplot(2, 1, 1), plt.imshow(matched, cmap='gray'), plt.axis('off')
# plt.title("Original Image")
plt.subplot(2, 1, 2), plt.imshow(img, cmap='gray'), plt.axis('off')
# plt.title("Original Image")
plt.show()
