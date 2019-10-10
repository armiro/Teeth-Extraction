import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
# from ROI_extractor import extract_roi

from preprocessing import CLAHE, match_histogram, draw_histogram

for i in range(1, 5):
    img = cv2.imread('./test-cropped-images/%d.bmp' % i, 0)
    # draw_histogram(image=img)
    unedited_img = copy.deepcopy(img)

    unedited_img = CLAHE(image=unedited_img, grid_size=8, clip_limit=2.)

    height, width = img.shape[0], img.shape[1]
    middle_width = int(width / 2.)
    # blur_window_size = int((height + width) / 7.)
    # if blur_window_size % 2 is 0:
    #     blur_window_size += 1
    #
    # img = cv2.GaussianBlur(img, (blur_window_size, blur_window_size), 0)
    print('image shape is:', img.shape)

    p1 = img[:, :int(5 * middle_width / 7.)]
    # cv2.line(img, (int(5 * middle_width / 7.), 0), (int(5 * middle_width / 7.), height), 0, 4)

    left_shadow = img[:, int(5 * middle_width / 7.):int(28 * middle_width / 30.)]
    # cv2.line(img, (int(29 * middle_width / 30.), 0), (int(29 * middle_width / 30.), height), 0, 4)

    p2 = img[:, int(28 * middle_width / 30.):int(32 * middle_width / 30.)]
    # cv2.line(img, (int(31 * middle_width / 30.), 0), (int(31 * middle_width / 30.), height), 0, 4)

    right_shadow = img[:, int(32 * middle_width / 30.):int(9 * middle_width / 7.)]
    # cv2.line(img, (int(9 * middle_width / 7.), 0), (int(9 * middle_width / 7.), height), 0, 4)

    p3 = img[:, int(9 * middle_width / 7.):]

    template_left = img[:, :int(width / 3.)]
    matched_left = match_histogram(source=left_shadow, template=template_left)
    template_right = img[:, int(2 * width / 3.):]
    matched_right = match_histogram(source=right_shadow, template=template_right)
    final = np.hstack((p1, matched_left, p2, matched_right, p3))
    final = CLAHE(image=final)

    plt.subplot(2, 1, 1), plt.imshow(final, cmap='gray'), plt.axis('off')
    plt.subplot(2, 1, 2), plt.imshow(unedited_img, cmap='gray'), plt.axis('off')
    plt.show()
