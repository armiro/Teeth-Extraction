import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
# from sklearn.cluster import KMeans
import preprocessing
from scipy.ndimage.morphology import binary_fill_holes


for i in range(3, 4):
    """load the image and do the pre-processing section"""
    img = cv2.imread('./test-manual/%d.bmp' % i, 0)
    height, width = img.shape[0], img.shape[1]
    # print(img.shape)
    img = preprocessing.CLAHE(image=img)
    # img = preprocessing.dilation(image=img, kernel_size=8, iterations=2, return_result=1)

    # img = cv2.GaussianBlur(img, (21, 21), 0)
    # it is the local CLAHE:
    # img[:, int(width/3.):int(width/1.5)] = preprocessing.CLAHE(image=img[:, int(width/3.):int(width/1.5)],
    #                                                            grid_size=2, clip_limit=1.)
    # note: savoula algorithm works much better without local CLAHE

    # desired_window = int(height / 15.) * 2 + 1
    # img = preprocessing.sauvola(image=img, window_size=desired_window, return_result=True)
    # img = np.array(img, dtype='uint8')
    # img = preprocessing.dilation(image=img, kernel_size=3, iterations=1, return_result=True)
    # img = binary_fill_holes(input=img)

    # img = preprocessing.erosion(img, 10, 3, 0, 1)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    """divide image using number of windows (n)"""
    n = 100
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

        # a = width/(n * (max(sum_array) - min(sum_array)))
        # b = -1 * a * min(sum_array)
        # for sum_idx in range(0, len(sum_array)):
        #     sum_array[sum_idx] = (a * sum_array[sum_idx]) + b

        # binarized = list()
        # for sum_element in sum_array:
        #     binarized.append(sum_element > 5)
        # plt.plot(binarized)
        # plt.show()

        tmp_array = copy.deepcopy(x=sum_array)
        # for e in range(0, len(tmp_array)):
        #     tmp_array[e] = (tmp_array[e] > 4000)

        # plt.plot(tmp_array)
        # plt.show()

        minimum = min(sum_array)
        minimum_idx = sum_array.index(minimum) + height_open

        tmp = int(width / n) * ii
        middle_dot = (int((2 * tmp + part.shape[1]) / 2.), minimum_idx)

        # cv2.line(img_with_dots, (tmp, minimum_idx), (tmp + part.shape[1], minimum_idx), 255, 3)
        cv2.circle(img=img_with_dots, center=middle_dot, radius=5, color=(255, 0, 0), thickness=-1)

        middle_dots.append(middle_dot)

    plt.imshow(img_with_dots, cmap='gray')
    plt.show()
    # print(middle_dots)

    """plot the middle points and fit it a graph"""
    # x = list()
    # y = list()
    # y_tmp = list()
    # for dot in middle_dots:
    #     x.append(dot[0])
    #     y.append(dot[1])
    #     y_tmp.append((dot[1]))
    #
    # z = np.polyfit(x=x, y=y_tmp, deg=2, rcond=None, full=False)
    # p = np.poly1d(z)
    #
    # xp = np.linspace(start=0, stop=width, num=width)
    # plt.plot(x, y_tmp, '.', xp, p(xp), '-')
    # plt.ylim(height, 0), plt.xlim(0, width)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # """now, let's remove the outliers"""
    # y_mean = np.mean(a=y)
    # y_std = np.std(a=y)
    # upper_bound = int(y_mean + y_std); lower_bound = int(y_mean - y_std)
    #
    # # print(upper_bound)
    # # print(lower_bound)
    # inliers = list()
    # for dot in middle_dots:
    #     if (dot[1] < upper_bound) & (dot[1] > lower_bound):
    #         inliers.append(dot)
    #
    # # print(inliers)
    # # print(middle_dots)
    #
    # x = list()
    # y = list()
    # for inlier in inliers:
    #     x.append(inlier[0])
    #     y.append((inlier[1]))
    #
    # z = np.polyfit(x=x, y=y, deg=2, rcond=None, full=False)
    # p = np.poly1d(z)
    #
    # xp = np.linspace(start=0, stop=width, num=width)
    # plt.imshow(img, cmap='gray')
    # plt.plot(x, y, '.', xp, p(xp), '-')
    # plt.ylim(height, 0), plt.xlim(0, width)
    # plt.show()

    """find the starting point"""
    starting_x = int(width / 2.)
    x_bound = int(width / 5.)
    mid_y = list()
    for point in middle_dots:
        if (point[0] > (starting_x - x_bound)) & (point[0] < (starting_x + x_bound)):
            mid_y.append(point[1])
    starting_y = int(np.mean(a=mid_y))
    supposed_starting_point = (starting_x, starting_y)
    # print(supposed_starting_point)
    # cv2.circle(img=img, center=starting_point, radius=5, color=(255, 0, 0), thickness=-1)

    # middle_dots = np.asarray(a=middle_dots)
    # tmp_2 = middle_dots[:, 1]
    # distances = np.sum(a=(tmp_2 - starting_y) ** 2, axis=1)
    distances = list()
    for point in middle_dots:
        if (point[0] > (starting_x - x_bound)) & (point[0] < (starting_x + x_bound)):
            distances.append((point[1] - starting_y) ** 2)
    # print(distances)
    nearest = np.argmin(a=distances)
    # print(nearest)
    poi_idx = int(nearest + int((starting_x - x_bound) / window_size))
    # print(middle_dots[poi])
    poi = middle_dots[poi_idx]
    # point_of_interest_y = tmp_2[np.argmin(a=distances)]
    # point_of_interest = middle_dots.index(point_of_interest_y)
    # cv2.circle(img=img, center=poi, radius=4, color=(255, 0, 0), thickness=-1)
    # plt.imshow(img, cmap='gray')
    # plt.show()


    def calculate_distance(point_1, point_2):
        dist = math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)
        return round(dist, 4)

    right_idx_vector = list([poi_idx])
    for idx in range(poi_idx+1, len(middle_dots)):
        distance = calculate_distance(point_1=middle_dots[idx], point_2=middle_dots[right_idx_vector[-1]])
        threshold = math.sqrt((1 + (idx - right_idx_vector[-1]) ** 2))
        threshold = threshold * window_size
        if (distance ** 0.9) <= threshold:
            right_idx_vector.append(idx)

    # print(right_idx_vector)

    for element in right_idx_vector:

        cv2.circle(img=img, center=middle_dots[element], radius=4, color=(255, 0, 0), thickness=-1)

    left_idx_vector = list([poi_idx])
    for idx in range(poi_idx - 1, 0, -1):
        distance = calculate_distance(point_1=middle_dots[idx], point_2=middle_dots[left_idx_vector[-1]])
        threshold = math.sqrt((1 + (idx - left_idx_vector[-1]) ** 2))
        threshold = threshold * window_size
        if (distance ** 0.9) <= threshold:
            left_idx_vector.append(idx)

    # print(right_idx_vector)

    for element in left_idx_vector:
        cv2.circle(img=img, center=middle_dots[element], radius=4, color=(255, 0, 0), thickness=-1)
    plt.imshow(img, cmap='gray')
    plt.show()
