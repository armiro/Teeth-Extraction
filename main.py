import cv2
import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from ROI_revision import *
from ROI_extraction import *
# from middle_line_via_points import draw_middle_line

for i in range(1, 51):
    img_address = './images/%d.bmp' % i
    img = cv2.imread(img_address, 0)
    print('original image dimensions:', img.shape)
    initial_roi, initial_boundaries = extract_roi(image=img, return_result=1)
    print('initial ROI dimensions:', initial_roi.shape)
    revised_roi, revised_boundaries = revise_boundaries(image=initial_roi, return_result=1)
    print('final ROI dimensions:', revised_roi.shape)
    # revised_roi = draw_middle_line(image=revised_roi, num_parts=60, show_result=False, return_result=True)

    # print(initial_boundaries)
    # print(final_boundaries)

    upper_height = initial_boundaries[3] + revised_boundaries[3]
    left_width = initial_boundaries[0] + revised_boundaries[0]
    lower_height = upper_height + revised_roi.shape[0]
    right_width = left_width + revised_roi.shape[1]

    top_left_corner = (left_width, upper_height)
    top_right_corner = (right_width, upper_height)
    bottom_left_corner = (left_width, lower_height)
    bottom_right_corner = (right_width, lower_height)

    print('roi points:', top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner)

    cv2.rectangle(img, top_left_corner, bottom_right_corner, 0, 7)
    # cv2.circle(img, top_left_corner, 20, 1, -1)
    # cv2.circle(img, top_right_corner, 20, 1, -1)
    # cv2.circle(img, bottom_left_corner, 20, 1, -1)
    # cv2.circle(img, bottom_right_corner, 20, 1, -1)

    # fig = plt.figure()
    # plt.subplot(2, 1, 1),
    plt.imshow(X=img, cmap='gray')
    # plt.subplot(2, 1, 2), plt.imshow(X=revised_roi, cmap='gray')
    plt.show()

    # file_name = './cropped-figures/%d' % i
    # fig.savefig(file_name)

    cv2.imwrite('cropped-images/%d.bmp' % i, img)
