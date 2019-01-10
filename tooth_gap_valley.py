import cv2
import numpy as np
import matplotlib.pyplot as plt
# from ROI_extractor import extract_roi

from preprocessing import CLAHE, global_threshold, erosion

img = cv2.imread('./test-cropped-images/4.bmp', 0)
img = CLAHE(image=img)
img = erosion(image=img, iterations=3, kernel_size=8, return_result=True)
# height, width = img.shape[0], img.shape[1]

lower_edge_detector = [[-1, -1, -1, -1, -1],
                       [0,  0,  0,  0,  0],
                       [0,  0,  0,  0,  0],
                       [0,  0,  0,  0,  0],
                       [1,  1,  1,  1,  1]]

# lower_edge_detector = [[-1, -1, -1, -1],
#                        [0,  0,  0,  0],
#                        [0,  0,  0,  0],
#                        [1,  1,  1,  1]]

# lower_edge_detector = [[-1, -1, -1],
#                        [0,  0,  0],
#                        [1,  1,  1]]

lower_kernel = np.array(lower_edge_detector, dtype=np.float32) / 1.0
lower_edges = cv2.filter2D(img, -1, lower_kernel)
lower_edges = cv2.GaussianBlur(lower_edges, (45, 45), 0)
lower_edges = global_threshold(image=lower_edges, threshold=64, return_result=True)


plt.subplot(2, 1, 1), plt.imshow(img, cmap='gray'), plt.axis('off')
plt.subplot(2, 1, 2), plt.imshow(lower_edges, cmap='gray'), plt.axis('off')
plt.show()
