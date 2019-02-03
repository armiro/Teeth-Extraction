import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocessing import CLAHE


def detect_left_boundary(image):
    # use a matrix as the kernel, convolve it on the image to find black-to-white vertical edges
    vertical_edge_detector = [[-2, 0, 2],
                              [-2, 0, 2],
                              [-2, 0, 2]]
    v_kernel = np.array(vertical_edge_detector, dtype=np.float32) / 1.0
    v_edges = cv2.filter2D(image, -1, v_kernel)

    # do a heavy bilateral filtering, since the image is too noisy
    v_edges = cv2.bilateralFilter(v_edges, 45, 75, 75)

    # extract the dimensions of the input image
    height, width = image.shape[0], image.shape[1]

    # pick a specific window (experimentally obtained) of the image as the "left_window"
    left_window_close = int(width / 5.0)
    left_window_open = int(width / 10.0)

    # find the average intensity of all the pixels in the left window
    left_quarter = v_edges[:, left_window_open:left_window_close]
    mean_intensity_left = float(round(left_quarter.mean(), 2))
    # print("average intensity of left-window is:", mean_intensity_left)

    # find the intensity of all the image columns residing inside the left window
    intensity_list = list(sum(v_edges[:, left_window_open:left_window_close]))

    # define initial value of required variables
    mx = mean_intensity_left
    possible_left_boundaries = list()

    # find the possible boundaries; the condition detects every dark-to-bright supervene change (edge)
    for idx in range(0, len(intensity_list) - 2):
        if (intensity_list[idx] < mx) & (intensity_list[idx + 1] > mx) & (intensity_list[idx + 2] > mx):
            possible_left_boundaries.append(idx)

    # if the condition is not met at all, just pass a rational left boundary to the list
    if possible_left_boundaries == []:
        possible_left_boundaries.append(int((left_window_open + left_window_close) / 3))

    # the last element is the best left boundary (resulting in smaller ROI image)
    left_boundary = possible_left_boundaries[-1] + left_window_open

    # draw the left boundary as a black line on the input image
    v_thickness = int(width / 300)
    adaptive_equalized = cv2.line(image, (left_boundary, 0), (left_boundary, height), 0, v_thickness)

    return left_boundary


def detect_right_boundary(image, left_boundary):
    # extract the width of the input image
    height, width = image.shape[0], image.shape[1]

    # the right boundary is symmetrical to the left one (in most cases)
    right_boundary = width - left_boundary

    # draw the right boundary as a black line on the input image
    v_thickness = int(width / 300)
    adaptive_equalized = cv2.line(image, (right_boundary, 0), (right_boundary, height), 0, v_thickness)

    return right_boundary


def detect_lower_boundary(image, left_boundary):
    # use a matrix as the kernel to find black-to-white horizontal edges
    horizontal_edge_detector = [[2, 2, 2],
                                [0, 0, 0],
                                [-2, -2, -2]]
    h_kernel = np.array(horizontal_edge_detector, dtype=np.float32) / 1.0
    h_edges = cv2.filter2D(image, -1, h_kernel)
    h_edges = cv2.bilateralFilter(h_edges, 45, 75, 75)

    # extract the dimensions of the input image
    height, width = image.shape[0], image.shape[1]

    # choose a specific window (experimentally obtained) as the "lower_window"
    lower_window_close = int(height / 2.5)
    lower_window_open = int(height / 5.0)

    # extract the column of the left boundary as "column_of_interest"
    column_of_interest = h_edges[:, left_boundary]

    # reverse it. because we want to find the outer edge of the maxilla, which is the lower boundary
    column_of_interest = list(reversed(column_of_interest))

    # find the intensity of all the image rows residing inside the lower window
    lower_quarter = h_edges[lower_window_open:lower_window_close, left_boundary]
    mean_intensity_lower = float(round(lower_quarter.mean(), 2))
    # print("average intensity of lower-window is:", mean_intensity_lower)

    # define initial value of required variables
    ml = mean_intensity_lower
    possible_lower_boundaries = list()

    # find the possible boundaries; the condition detects every dark-to-bright supervene change (edge)
    for idx in range(lower_window_open, lower_window_close - 2):
        if (column_of_interest[idx] < ml) & (column_of_interest[idx + 1] >= ml) & (column_of_interest[idx + 2] >= ml):
            possible_lower_boundaries.append(idx)

    # if the condition is not met at all, just pass a rational lower boundary to the list
    if possible_lower_boundaries == []:
        possible_lower_boundaries.append(int((lower_window_open + lower_window_close) / 3))

    # the first element is the best lower boundary (resulting in smaller ROI image)
    lower_boundary = height - possible_lower_boundaries[0]

    # draw the lower boundary as a black line on the input image
    h_thickness = int(height / 150)
    adaptive_equalized = cv2.line(image, (0, lower_boundary), (width, lower_boundary), 0, h_thickness)

    return lower_boundary


def detect_upper_boundary(image, left_boundary):
    # use a matrix as the kernel to find white-to-black horizontal edges
    horizontal_edge_detector = [[-2, -2, -2, -2, -2],
                                [0,  0,  0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [2,  2,  2, 2, 2]]
    h_kernel = np.array(horizontal_edge_detector, dtype=np.float32) / 1.0
    h_edges = cv2.filter2D(image, -1, h_kernel)
    h_edges = cv2.bilateralFilter(h_edges, 45, 75, 75)

    # extract the dimensions of the input image
    height, width = image.shape[0], image.shape[1]

    # choose a specific window (experimentally obtained) as the "lower_window"
    upper_window_close = int(height / 2.0)
    upper_window_open = int(height / 5.0)

    # extract the column of the left boundary as "column_of_interest"
    column_of_interest = h_edges[:, left_boundary]

    # find the intensity of all the image pixels residing inside the upper window of our column of interest
    upper_quarter = column_of_interest[upper_window_open:upper_window_close]
    mean_intensity_upper = float(round(upper_quarter.mean(), 2))
    # print("average intensity of upper-window is:", mean_intensity_upper)

    # define initial value of required variables
    mp = mean_intensity_upper
    possible_upper_boundaries = list()

    # find the possible boundaries; the condition detects every dark-to-bright supervene change (edge)
    for idx in range(upper_window_open, upper_window_close - 2):
        if (column_of_interest[idx] < mp) & (column_of_interest[idx + 1] >= mp) & (column_of_interest[idx + 2] >= mp):
            possible_upper_boundaries.append(idx)

    # if the condition is not met at all, just pass a rational lower boundary to the list
    if possible_upper_boundaries == []:
        possible_upper_boundaries.append(int((upper_window_open + upper_window_close) / 3))

    # the first element is the best lower boundary (resulting in smaller ROI image)
    upper_boundary = possible_upper_boundaries[0]
    # draw the lower boundary as a black line on the input image
    h_thickness = int(height / 150)
    adaptive_equalized = cv2.line(image, (0, upper_boundary), (width, upper_boundary), 0, h_thickness)

    return upper_boundary


def extract_roi(image, return_result=False, show_result=False):
    # at first, load the image and do the CLAHE as the pre-processing step
    # print("image dimensions are:", image.shape)
    adaptive_equalized = CLAHE(image=image, clip_limit=2.0, grid_size=8)

    # then, extract the desired boundaries
    left_boundary = detect_left_boundary(image=adaptive_equalized)
    right_boundary = detect_right_boundary(image=adaptive_equalized, left_boundary=left_boundary)
    lower_boundary = detect_lower_boundary(image=adaptive_equalized, left_boundary=left_boundary)
    upper_boundary = detect_upper_boundary(image=adaptive_equalized, left_boundary=left_boundary)

    roi = image[upper_boundary:lower_boundary, left_boundary:right_boundary]
    boundaries = [left_boundary, right_boundary, lower_boundary, upper_boundary]
    # plot the result as well as the input image
    if show_result:
        plt.subplot(211), plt.imshow(adaptive_equalized, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(roi, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return roi, boundaries

