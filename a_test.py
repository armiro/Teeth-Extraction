import cv2
import numpy as np
import matplotlib.pyplot as plt

import preprocessing
from scipy.ndimage.morphology import binary_fill_holes
# import clustering


def divide_image(image, n):
    # height, width = image.shape[0], img.shape[1]
    window_size = int(width/n)
    parts = list()
    for part_idx in range(0, n):
        parts.append(image[:, (window_size*part_idx):(window_size*part_idx + window_size)])

    # plt.imshow(parts[3], cmap='gray')
    # plt.show()
    ii = -1
    for part in parts:
        ii += 1
        this_part = np.array(part, dtype='uint8')
        # this_part = cv2.GaussianBlur(this_part, (41, 41), 0)
        # this_part = preprocessing.dilation(image=this_part, kernel_size=3, iterations=2, return_result=True)

        # this_part = binary_fill_holes(input=this_part)
        # plt.imshow(this_part, cmap='gray')
        # plt.show()

        sum_array = list()
        for line_idx in range(0, height):
            line = this_part[line_idx, :]
            sum_array.append(sum(line))

        # a = width/(n * (max(sum_array) - min(sum_array)))
        # b = -1 * a * min(sum_array)
        # for sum_idx in range(0, len(sum_array)):
        #     sum_array[sum_idx] = (a * sum_array[sum_idx]) + b


        # binarized = list()
        # for sum_element in sum_array:
        #     binarized.append(sum_element > 5)
        # plt.plot(binarized)
        # plt.show()

        minimum = min(sum_array)
        minimum_idx = sum_array.index(minimum)
        # plt.imshow(this_part)
        # plt.show()

        tmp = int(width/n) * ii
        cv2.line(img, (tmp, minimum_idx), (tmp + this_part.shape[1], minimum_idx), (255, 0, 0), 2)
    plt.imshow(img)
    plt.show()


for i in range(1, 2):
    img = cv2.imread('./test-images/test.bmp', 0)
    height, width = img.shape[0], img.shape[1]
    # print(img.shape)
    desired_window = int(height/15.)*2 + 1
    img = preprocessing.CLAHE(image=img)
    # mid_window = img[:, int(width/3.):int(width/1.5)]
    img[:, int(width/3.):int(width/1.5)] = preprocessing.CLAHE(image=img[:, int(width/3.):int(width/1.5)],
                                                               grid_size=12)
    # img[:, 123:245] = preprocessing.CLAHE(image=img[:, 123:245], grid_size=12)
    # print(desired_window)
    img = preprocessing.sauvola(image=img, window_size=desired_window, return_result=True)
    img = np.array(img, dtype='uint8')
    img = preprocessing.dilation(image=img, kernel_size=3, iterations=1, return_result=True)
    # img = binary_fill_holes(input=img)
    plt.imshow(img, cmap='gray')
    plt.show()


divide_image(image=img, n=int(width/5))
