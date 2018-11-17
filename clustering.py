from preprocessing import *


def kmeans(image, num_clusters, image_depth):
    z = image.reshape((-1, image_depth))
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(z, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)
    return res2


clustering1 = kmeans(image=img, num_clusters=2, image_depth=1)
clustering2 = kmeans(image=equalized, num_clusters=2, image_depth=1)
clustering3 = kmeans(image=adaptive_equalized, num_clusters=2, image_depth=1)

# cv2.imshow('kmeans on raw image', clustering1)
# cv2.waitKey(0)
# cv2.imshow('kmeans on histogram equalized image', clustering2)
# cv2.waitKey(0)
# cv2.imshow('kmeans on adaptive histogram equalized image', clustering3)
# cv2.waitKey(0)


plt.subplot(2, 1, 1)
plt.imshow(clustering1, cmap='gray')
plt.subplot(2, 1, 2)
plt.imshow(clustering3, cmap='gray')
plt.show()

