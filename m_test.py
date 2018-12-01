import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./test-images/source.jpg', 0)

retVal = cv2.ximgproc.createRFFeatureGetter()
res = cv2.ximgproc.createStructuredEdgeDetection()


for l in range(5):
    cmap = plt.cm.get_cmap("Spectral")
    plt.contour(label == l, contours=1, colors=[cmap(l / float(n_clusters)), ])
plt.axis('off')
plt.show()
# plt.subplot(2, 1, 1), plt.imshow(img, cmap='gray'), plt.axis('off')
# plt.subplot(2, 1, 2), plt.imshow(retVal, cmap='gray'), plt.axis('off')
# plt.show()
