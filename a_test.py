import cv2
import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import copy
import clustering

for i in range(3, 10):
    img_address = './test-cropped-images/%d.bmp' % i
    img = cv2.imread(img_address, 0)
    height, width = img.shape[0], img.shape[1]
    print(img.shape)
    # img = preprocessing.CLAHE(image=img)
    tmp = copy.deepcopy(x=img)
    desired_kernel_size = int(width / 80) * 2 + 1
    tmp = cv2.GaussianBlur(tmp, (desired_kernel_size, desired_kernel_size), 0)
    desired_window_size = int(width/60)*2 + 1
    tmp = preprocessing.sauvola(image=tmp, window_size=desired_window_size, return_result=1)
    tmp = np.array(tmp, dtype='uint8')
    tmp = cv2.GaussianBlur(tmp, (desired_kernel_size, desired_kernel_size), 0)

    sum_array = list()
    bound = int(height / 5)
    # bound = 0
    for line_idx in range(bound, height-bound):
        line = tmp[line_idx, :]
        sum_array.append(sum(line))

    # print(sum_array)
    peak = np.argmin(a=sum_array) + bound
    # print(peak)
    # cv2.line(img, (0, peak), (width, peak), 0, 3)
    # plt.imshow(X=img, cmap='gray')
    # plt.plot(sum_array)
    # plt.show()

    # these coefficient are obtained empirically
    # upper is between 0.4 and 0.5
    # lower is between 0.3 and 0.4
    upper_bound = peak - int(0.45 * height)
    lower_bound = peak + int(0.4 * height)

    if upper_bound < 0:
        upper_bound = 0
    if lower_bound > height:
        lower_bound = height

    img = img[upper_bound:lower_bound, :]
    # plt.imshow(X=img, cmap='gray')
    # plt.show()

    """test methods in order to detect vertical edges correctly"""
    img = preprocessing.CLAHE(image=img)

    # preprocessing.quadtree_decomp(image=img, min_size=size, min_std=std, show_result=True)
    binarized = preprocessing.sauvola(image=img, window_size=125, show_result=0, return_result=1)
    binarized = np.array(binarized, dtype='uint8')
    # binarized = preprocessing.niblack(image=img, window_size=205, k=1, show_result=1, return_result=1)
    # binarized = np.array(binarized, dtype='uint8')
    # binarized = preprocessing.otsu(image=img, blur=1, show_result=1, return_result=1)

    binarized = preprocessing.dilation(image=binarized, kernel_size=5, iterations=2, return_result=1, show_result=1)

    # clustering.hierarchical(image=binarized, num_clusters=2)

