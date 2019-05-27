# import cv2
import preprocessing as prep
import numpy as np


def separate_jaws(image):
    height, width = image.shape[:2]
    upper_jaw = np.zeros(shape=(height, 1))
    lower_jaw = np.zeros(shape=(height, 1))
    line_points = list()

    # change pixels with the value of 0 to 1; for further purpose of applying genetic algorithm
    image = prep.eliminate_black_pixels(image=image)
    middle_line_pixels = list()

    for column in range(0, width):
        this_column = image[:, column]
        for px_idx in range(0, len(this_column)):
            if this_column[px_idx] == 255:
                line_points.append([column, px_idx])
                up = np.vstack(tup=(image[:px_idx, column:column+1], np.zeros(shape=(height-px_idx, 1))))
                upper_jaw = np.hstack(tup=(upper_jaw, up))

                dn = np.vstack(tup=(np.zeros(shape=(px_idx+1, 1)), image[px_idx+1:, column:column+1]))
                lower_jaw = np.hstack(tup=(lower_jaw, dn))

                middle_line_pixels.append(px_idx)
                break

    upper_bound = max(middle_line_pixels)
    upper_jaw = upper_jaw[:upper_bound + 1, :]

    lower_bound = min(middle_line_pixels)
    lower_jaw = lower_jaw[lower_bound + 1:, :]

    return upper_jaw, lower_jaw

