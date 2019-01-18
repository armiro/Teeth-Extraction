import cv2
import numpy as np
import matplotlib.pyplot as plt
# from ROI_extractor import extract_roi

import preprocessing

img = cv2.imread('./test-cropped-images/1.bmp', 0)
print(img.shape)
# img = CLAHE(image=img)
# img = erosion(image=img, iterations=3, kernel_size=8, return_result=True)
nblk = preprocessing.niblack(image=img, window_size=29, k=0, show_result=False, return_result=True)
# nblk = preprocessing.global_threshold(image=img, threshold=127, return_result=True)
nblk = np.array(nblk, dtype='uint8')

# floodfill

# h, w = img.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)
# cv2.floodFill(nblk, None, (0, 0), 255)
# print(nblk)

# nblk = preprocessing.dilation(image=nblk, kernel_size=2, iterations=2, return_result=True)
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

# lower_kernel = np.array(lower_edge_detector, dtype=np.float32) / 1.0
# lower_edges = cv2.filter2D(img, -1, lower_kernel)
# lower_edges = cv2.GaussianBlur(lower_edges, (7, 7), 0)
# test = global_threshold(image=img, threshold=127, return_result=True)

# minLineLength = 100
# maxLineGap = 0
# lines = cv2.HoughLinesP(nblk, 1, np.pi/180, 100, minLineLength, maxLineGap)
# print(lines)
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(nblk,(x1,y1),(x2,y2),(0,0,0),1)

# plt.subplot(2, 1, 1), plt.imshow(img), plt.axis('off')
# plt.subplot(2, 1, 2), plt.imshow(lower_edges), plt.axis('off')
# plt.show()
plt.imshow(nblk, cmap='gray')
plt.show()