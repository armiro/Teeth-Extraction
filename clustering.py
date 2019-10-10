import time as time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering


def kmeans(image, num_clusters):
    z = image.reshape((-1, 1))
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(z, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)

    plt.subplot(2, 1, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(2, 1, 2), plt.imshow(res2, cmap='gray')
    plt.title('KMeans with %d clusters' % num_clusters), plt.axis('off')
    plt.show()


def hierarchical(image, num_clusters):
    x = np.reshape(image, (-1, 1))
    # Define the structure A of the data. Pixels connected to their neighbors
    connectivity = grid_to_graph(*image.shape)

    # Compute clustering
    print("Compute structured hierarchical clustering...")
    elapsed_time = time.time()
    ward = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward', connectivity=connectivity).fit(x)
    label = np.reshape(ward.labels_, image.shape)
    print("Elapsed time (sec): ", round(time.time() - elapsed_time, 2))
    print("Number of pixels: ", label.size)

    # Plot the results on an image
    plt.imshow(image, cmap='gray')
    # gray = label.sum(-1)/n_clusters
    for l in range(num_clusters):
        cmap = plt.cm.get_cmap("Spectral")
        plt.contour(label == l, colors=[cmap(l / float(num_clusters)), ])
    plt.title('Hierarchical with %d clusters' % num_clusters)
    plt.axis('off')
    plt.show()

