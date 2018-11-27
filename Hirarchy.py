# Author : Vincent Michel, 2010
#          Alexandre Gramfort, 2011
# License: BSD 3 clause



import time as time
import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

###############################################################################
# Generate data
lena = cv2.imread('5.jpg',0)
equ = cv2.equalizeHist(lena)
lena = equ
# Downsample the image by a factor of 4
#lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
X = np.reshape(lena, (-1, 1))

###############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*lena.shape)

###############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 30  # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters,
        linkage='ward', connectivity=connectivity).fit(X)
label = np.reshape(ward.labels_, lena.shape)
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)

###############################################################################
# Plot the results on an image

plt.figure(figsize=(5, 5))
plt.imshow(lena, cmap=plt.cm.gray)
#gray = label.sum(-1)/n_clusters
for l in range(n_clusters):
    cmap = plt.cm.get_cmap("Spectral")
    plt.contour(label == l, contours=1, colors=[cmap(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
plt.show()
