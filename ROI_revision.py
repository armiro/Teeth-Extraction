import cv2
import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import copy


def revise_horizontal_boundaries(image, show_result=False, return_result=False):
    height, width = image.shape[0], image.shape[1]
    print("image dimensions are:", image.shape)
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
        return image


def revise_vertical_boundaries(image, show_result=False, return_result=False):
    pass
