import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import CLAHE

# first load the image and do the CLAHE as the pre-processing step
img_address = "./test-images/female_35.bmp"
img = cv2.imread(img_address, 0)
adaptive_equalized = CLAHE(image=img, clip_limit=2.0, grid_size=8)

# use a matrix as the kernel and convolve it on the image
# to find black-to-white vertical edges
vertical_edge_detector = [[-2, 0, 2],
                          [-2, 0, 2],
                          [-2, 0, 2]]
# another one to find black-to-white horizontal edges
horizontal_edge_detector = [[2,  2,  2],
                            [0,  0,  0],
                            [-2, -2, -2]]

v_kernel = np.array(vertical_edge_detector, dtype=np.float32) / 1.0
h_kernel = np.array(horizontal_edge_detector, dtype=np.float32) / 1.0
v_edges = cv2.filter2D(adaptive_equalized, -1, v_kernel)
h_edges = cv2.filter2D(adaptive_equalized, -1, h_kernel)
v_edges = cv2.bilateralFilter(v_edges, 17, 45, 45)
h_edges = cv2.bilateralFilter(h_edges, 17, 45, 45)

# "filtered" variable has the dimensions of input image
height, width = img.shape[0], img.shape[1]

# pick a specific window (arbitrarily) of the image as the "left_window"
left_window_close = int(width / 5.0)
left_window_open = int(width / 10.0)

lower_window_close = int(height / 5.0)
lower_window_open = int(height / 10.0)

# find the mean intensity of all the pixels in the left window of the image
left_quarter = v_edges[:, left_window_open:left_window_close]
mean_intensity_left = float(round(left_quarter.mean(), 2))
print("average intensity of left-window is:", mean_intensity_left)

# find the intensity of all the image columns residing inside the left window
intensity_list = list(sum(v_edges[:, left_window_open:left_window_close]))

# extend the list in order to prevent conflicts inside the "for" loop
intensity_list.extend([0.0, 0.0, 0.0])

# define initial value of required variables
mx = mean_intensity_left
possible_left_boundaries = list()

# find the possible boundaries; the condition detects every dark-to-bright supervene change (edge)
for idx in range(0, len(intensity_list) - 3):
    if (intensity_list[idx] < mx) & (intensity_list[idx + 1] > mx) & (intensity_list[idx + 2] > mx) & (intensity_list[idx + 3] > mx):
        possible_left_boundaries.append(idx)

# the last element is the best left boundary (resulting in smaller ROI image)
left_boundary = possible_left_boundaries[-1] + left_window_open
right_boundary = width - left_boundary

# draw the left boundary as a black line on the input image
adaptive_equalized = cv2.line(adaptive_equalized, (left_boundary, 0), (left_boundary, height), 0, 5)
adaptive_equalized = cv2.line(adaptive_equalized, (right_boundary, 0), (right_boundary, height), 0, 5)

column_of_interest = h_edges[:, left_boundary + 1]
column_of_interest = list(reversed(column_of_interest))
# print(column_of_interest)
possible_lower_boundaries = list()
for idx in range(0, height-1):
    if column_of_interest[idx] > mx:
        possible_lower_boundaries.append(idx)

lower_boundary = height - possible_lower_boundaries[0]
upper_boundary = height - lower_boundary + int(height / 9.)
print(possible_lower_boundaries)
adaptive_equalized = cv2.line(adaptive_equalized, (0, lower_boundary), (width, lower_boundary), 0, 5)
adaptive_equalized = cv2.line(adaptive_equalized, (0, upper_boundary), (width, upper_boundary), 0, 5)

# plot the result as well as the filtered image
plt.subplot(211), plt.imshow(adaptive_equalized, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(h_edges, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
