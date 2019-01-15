import cv2
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import preprocessing
from scipy.ndimage.morphology import binary_fill_holes


for i in range(1, 2):
    """load the image and do the pre-processing section"""
    img = cv2.imread('./test-cropped-images/%d.bmp' % i, 0)
    height, width = img.shape[0], img.shape[1]
    # print(img.shape)
    img = preprocessing.CLAHE(image=img)
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
    n = 50
    window_size = int(width / n)
    parts = list()
    middle_dots = list()
    height_open = int(height / 8.)
    height_close = height - height_open

    for part_idx in range(0, n):
        parts.append(img[height_open:height_close, (window_size * part_idx):(window_size * part_idx + window_size)])

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

        minimum = min(sum_array)
        minimum_idx = sum_array.index(minimum) + height_open

        tmp = int(width / n) * ii
        cv2.line(img, (tmp, minimum_idx), (tmp + part.shape[1], minimum_idx), 255, 3)
        middle_dots.append((int((2 * tmp + part.shape[1]) / 2.), minimum_idx))

    plt.imshow(img, cmap='gray')
    plt.show()
    # print(middle_dots)

    """plot the middle points and fit it a graph"""
    x = list()
    y = list()
    y_tmp = list()
    for dot in middle_dots:
        x.append(dot[0])
        y.append(dot[1])
        y_tmp.append((-1 * dot[1]))

    z = np.polyfit(x=x, y=y_tmp, deg=5, rcond=None, full=False)
    p = np.poly1d(z)

    xp = np.linspace(start=0, stop=width, num=width)
    plt.plot(x, y_tmp, '.', xp, p(xp), '-')
    plt.ylim((-1 * height), 0), plt.xlim(0, width)
    plt.show()

    """now, let's remove the outliers"""
    y_mean = np.mean(a=y)
    y_std = np.std(a=y)
    upper_bound = int(y_mean + y_std); lower_bound = int(y_mean - y_std)

    # print(upper_bound)
    # print(lower_bound)
    inliers = list()
    for dot in middle_dots:
        if (dot[1] < upper_bound) & (dot[1] > lower_bound):
            inliers.append(dot)

    # print(inliers)
    # print(middle_dots)

    xx = list()
    yy = list()
    for inlier in inliers:
        xx.append(inlier[0])
        yy.append((-1 * inlier[1]))

    z = np.polyfit(x=xx, y=yy, deg=5, rcond=None, full=False)
    p = np.poly1d(z)

    xp = np.linspace(start=0, stop=width, num=width)
    plt.plot(xx, yy, '.', xp, p(xp), '-')
    plt.ylim((-1 * height), 0), plt.xlim(0, width)
    plt.show()

