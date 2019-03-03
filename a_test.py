import cv2
import matplotlib.pyplot as plt
import preprocessing as prep
import numpy as np
import copy
from skimage.filters import rank
from skimage.morphology import disk

img_address = './test-auto-cropped/2.bmp'
# img_address = './test-images/up_jaw.bmp'
img = cv2.imread(img_address, 0)
h, w = img.shape[:2]
org = copy.deepcopy(x=img)

# img = rank.equalize(image=img, selem=disk(150))
# img = prep.CLAHE(image=img, clip_limit=2., grid_size=8)
# img = prep.equalize_histogram(image=img)
# img = prep.erosion(image=img, kernel_size=10, iterations=1, return_result=1)
# img = prep.dilation(image=img, kernel_size=6, iterations=1, return_result=1)

# img = prep.sauvola(image=img, window_size=175, show_result=False, return_result=True)
# img = np.array(img, dtype='uint8')
# img = prep.erosion(image=img, kernel_size=10, iterations=1, return_result=1)
# img = prep.imfill(image=img, threshold=10, return_result=1)
# img = np.array(img, dtype='uint8')

# cv2.imwrite('./test-images/2.bmp', img)
plt.subplot(2, 1, 1), plt.imshow(X=org, cmap='gray'), plt.axis('off')
plt.subplot(2, 1, 2), plt.imshow(X=img, cmap='gray'), plt.axis('off')
plt.show()


