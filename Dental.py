import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('4.jpg',0)
img = cv2.equalizeHist(img)
Z = img.reshape((-1,1))
r=len(img)
c=len(img[1,:])
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 6
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

label2 = label.reshape(r,c)

plt.imshow(res2)


#segmentation
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap=plt.cm.gray)
#gray = label.sum(-1)/n_clusters
n_clusters = K
for l in range(n_clusters):
    cmap = plt.cm.get_cmap("Spectral")
    plt.contour(label2 == l, contours=1, colors=[cmap(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
plt.show()
