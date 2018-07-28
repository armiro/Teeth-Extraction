import math
import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt

# from scipy.special import ellipk
# from sklearn.feature_extraction import image
# from sklearn.cluster import spectral_clustering
from skimage import io, filters
from skimage.color import rgb2gray
from skimage.segmentation import active_contour

path = "./test-images/female_27.bmp"
img = io.imread(path)
img = rgb2gray(img)
# img = filters.frangi(img, scale_range=(1, 15), scale_step=2,
#                      beta1=0.5, beta2=15, black_ridges=True)
io.imshow(img)
io.show()

# using Snake active contour for segmentation
s = np.linspace(0, 2*math.pi, 100)
x = img.shape[1]/2 + 180*np.sin(s)
y = img.shape[0]/2 + 180*np.cos(s)
init = np.array([x, y]).T

# snake = active_contour(img, init, bc='periodic',
#                        alpha=0.015, beta=10, gamma=0.001)
#
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
# ax.plot(snake[:, 0], snake[:, 1], '-b', lw=2)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
# plt.show()

