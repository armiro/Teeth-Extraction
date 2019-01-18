import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import preprocessing
from scipy.ndimage.morphology import binary_fill_holes


for i in range(2, 3):
    """load the image and do the pre-processing section"""
    img = cv2.imread('./test-manual/%d.bmp' % i, 0)
    height, width = img.shape[0], img.shape[1]
    # print(img.shape)
    img = preprocessing.CLAHE(image=img)

    """divide image using number of windows (n)"""
    n = 50
    window_size = int(width / n)
    parts = list()
    middle_dots = list()
    height_open = int(height / 8.)
    height_close = height - height_open

    for part_idx in range(0, n):
        parts.append(img[height_open:height_close, (window_size * part_idx):(window_size * part_idx + window_size)])

    img_with_dots = copy.deepcopy(x=img)
    ii = -1
    for part in parts:
        ii += 1
        sum_array = list()

        for line_idx in range(0, len(part)):
            line = part[line_idx, :]
            sum_array.append(sum(line))

        minimum = min(sum_array)
        minimum_idx = sum_array.index(minimum) + height_open

        tmp = int(width / n) * ii
        middle_dot = (int((2 * tmp + part.shape[1]) / 2.), minimum_idx)

        # cv2.line(img_with_dots, (tmp, minimum_idx), (tmp + part.shape[1], minimum_idx), 255, 3)
        cv2.circle(img=img_with_dots, center=middle_dot, radius=5, color=(255, 0, 0), thickness=-1)

        middle_dots.append(middle_dot)

    # plt.imshow(img_with_dots, cmap='gray')
    # plt.show()
    # print(middle_dots)

    """find the starting point"""
    starting_x = int(width / 2.)
    x_bound = int(width / 5.)
    mid_y = list()
    for point in middle_dots:
        if (point[0] > (starting_x - x_bound)) & (point[0] < (starting_x + x_bound)):
            mid_y.append(point[1])
    starting_y = int(np.mean(a=mid_y))
    # supposed_starting_point = (starting_x, starting_y)
    # print(supposed_starting_point)
    # cv2.circle(img=img, center=starting_point, radius=4, color=255, thickness=-1)

    distances = list()
    for point in middle_dots:
        if (point[0] > (starting_x - x_bound)) & (point[0] < (starting_x + x_bound)):
            distances.append((point[1] - starting_y) ** 2)

    # print(distances)
    nearest = np.argmin(a=distances)
    # print(nearest)

    poi_idx = int(nearest + int((starting_x - x_bound) / window_size))
    # print(poi_idx)
    poi = middle_dots[poi_idx]

    right_points_y = list([poi[1]])
    for idx in range(poi_idx, len(middle_dots)-1):
        this_y = right_points_y[-1]
        tmp_var = window_size
        next_window = img[this_y - tmp_var:this_y + tmp_var, (idx * (window_size+1)): (idx * (window_size+1))+1]
        next_poi_idx = np.argmin(np.sum(a=next_window, axis=1))
        next_poi_y = this_y - window_size + next_poi_idx
        # print(next_window)
        right_points_y.append(next_poi_y)
        cv2.circle(img=img, center=(idx*window_size, this_y), radius=4, color=(255, 0, 0), thickness=-1)

    plt.imshow(img, cmap='gray')
    plt.show()
    # print(right_idx_vector)
