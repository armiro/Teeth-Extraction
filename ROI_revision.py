import cv2
import preprocessing
import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
import numpy as np
import copy


def revise_horizontal_boundaries(image, show_result=False, return_result=False):
    height, width = image.shape[0], image.shape[1]
    # print("image dimensions are:", image.shape)
    # img = preprocessing.CLAHE(image=img)
    tmp = copy.deepcopy(x=image)
    desired_kernel_size = int(width / 80) * 2 + 1
    tmp = cv2.GaussianBlur(tmp, (desired_kernel_size, desired_kernel_size), 0)
    desired_window_size = int(width / 60) * 2 + 1
    tmp = preprocessing.sauvola(image=tmp, window_size=desired_window_size, return_result=1)
    tmp = np.array(tmp, dtype='uint8')
    tmp = cv2.GaussianBlur(tmp, (desired_kernel_size, desired_kernel_size), 0)

    sum_array = list()
    bound = int(height / 5)
    # bound = 0
    for line_idx in range(bound, height - bound):
        line = tmp[line_idx, :]
        sum_array.append(sum(line))

    peak = np.argmin(a=sum_array) + bound
    cv2.line(tmp, (0, peak), (width, peak), 0, 3)

    # these coefficients are obtained empirically
    # upper is between 0.4 and 0.5
    # lower is between 0.3 and 0.4
    upper_bound = peak - int(0.45 * height)
    lower_bound = peak + int(0.4 * height)

    if upper_bound < 0:
        upper_bound = 0
    if lower_bound > height:
        lower_bound = height

    image = image[upper_bound:lower_bound, :]

    if show_result:
        plt.subplot(2, 1, 1), plt.imshow(X=tmp, cmap='gray')
        plt.subplot(2, 1, 2), plt.imshow(X=image, cmap='gray')
        plt.show()

    if return_result:
        return image, lower_bound, upper_bound


def revise_vertical_boundaries(image, show_result=False, return_result=False):

    def find_eoi(image, h_start, h_stop, rev_count):
        sum_array = list()
        for line_idx in range(resized_width):
            line = image[h_start:h_stop, line_idx]
            sum_array.append(sum(line))

        mean = int(np.mean(a=sum_array))

        # ln = mlines.Line2D(xdata=[0, resized_width], ydata=[mean, mean], linewidth=2)
        # ax = plt.gca()
        # ax.add_line(ln)
        # plt.plot(sum_array)
        # plt.show()

        if rev_count:
            for element in range(len(sum_array) - 2, 0, -1):
                if (sum_array[element + 1] <= mean) & (sum_array[element] >= mean):
                    eoi = int(element * width / 500)
                    break
        else:
            for element in range(0, len(sum_array) - 2):
                if (sum_array[element] <= mean) & (sum_array[element + 1] >= mean):
                    eoi = int(element * width / 500)
                    break

        return eoi

    tmp = copy.deepcopy(x=image)
    tmp = preprocessing.CLAHE(image=tmp, clip_limit=2., grid_size=8)
    height, width = tmp.shape[0], tmp.shape[1]
    # print("image shape is:", tmp.shape)
    resized_img = cv2.resize(tmp, (0, 0), fx=(500 / width), fy=(200 / height))

    resized_height, resized_width = resized_img.shape[0], resized_img.shape[1]
    linewidth = int(width / 400.)

    # img = cv2.bilateralFilter(img, 17, 35, 35)
    # img = cv2.blur(img, (15, 15))
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.medianBlur(img, 15)

    """vertical edges filter: left boundary"""
    vertical_edge_detector = np.zeros(shape=(3, 3))
    for ii in range(0, len(vertical_edge_detector)):
        vertical_edge_detector[ii, 0] = -1
        vertical_edge_detector[ii, -1] = 1

    # print(vertical_edge_detector)
    v_kernel = np.array(vertical_edge_detector, dtype=np.float32) / 1.0
    left_edge = cv2.filter2D(resized_img, -1, v_kernel)
    """end"""

    left_edge = preprocessing.dilation(image=left_edge, kernel_size=10, iterations=2, return_result=1)
    # img = cv2.blur(img, (15, 15))

    left_eoi_ln = find_eoi(image=left_edge, h_start=0, h_stop=height, rev_count=False)
    left_eoi_lw = find_eoi(image=left_edge, h_start=int(resized_height / 2), h_stop=height, rev_count=False)
    left_eoi_up = find_eoi(image=left_edge, h_start=0, h_stop=int(resized_height / 2), rev_count=False)
    left_eoi = min([left_eoi_ln, left_eoi_lw, left_eoi_up])

    cv2.line(tmp, (left_eoi, 0), (left_eoi, height), 0, linewidth)
    # cv2.line(resized_img, (sum_element, 0), (sum_element, resized_height), 0, linewidth)
    # print("left eoi is:", left_eoi)

    """vertical edges filter: right boundary"""
    vertical_edge_detector = np.zeros(shape=(3, 3))
    for ii in range(0, len(vertical_edge_detector)):
        vertical_edge_detector[ii, 0] = 1
        vertical_edge_detector[ii, -1] = -1

    # print(vertical_edge_detector)

    v_kernel = np.array(vertical_edge_detector, dtype=np.float32) / 1.0
    right_edge = cv2.filter2D(resized_img, -1, v_kernel)
    """end"""

    right_edge = preprocessing.dilation(image=right_edge, kernel_size=10, iterations=2, return_result=1)
    # img = cv2.blur(img, (15, 15))

    right_eoi_ln = find_eoi(image=right_edge, h_start=0, h_stop=height, rev_count=True)
    right_eoi_lw = find_eoi(image=right_edge, h_start=int(resized_height / 2), h_stop=height, rev_count=True)
    right_eoi_up = find_eoi(image=right_edge, h_start=0, h_stop=int(resized_height / 2), rev_count=True)
    right_eoi = max([right_eoi_ln, right_eoi_lw, right_eoi_up])

    cv2.line(tmp, (right_eoi, 0), (right_eoi, height), 0, linewidth)
    # cv2.line(img_resized, (sum_element, 0), (sum_element, resized_height), 0, linewidth)
    # print("right eoi is:", right_eoi)

    image = image[:, left_eoi:right_eoi]

    if show_result:
        plt.subplot(2, 1, 1), plt.imshow(X=tmp, cmap='gray')
        plt.subplot(2, 1, 2), plt.imshow(X=image, cmap='gray')
        plt.show()

    if return_result:
        return image, left_eoi, right_eoi


def revise_boundaries(image, show_result=False, return_result=False):

    h_rev, lower_boundary, upper_boundary = revise_horizontal_boundaries(image=image, return_result=True)
    revised_roi, left_boundary, right_boundary = revise_vertical_boundaries(image=h_rev, return_result=True)
    boundaries = [left_boundary, right_boundary, lower_boundary, upper_boundary]

    # plot the result as well as the input image
    if show_result:
        plt.subplot(211), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(revised_roi, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return revised_roi, boundaries


