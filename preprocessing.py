import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_histogram(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


def equalize_histogram(image):
    equ = cv2.equalizeHist(image)
    res = np.hstack((image, equ))
    return equ, res


def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(image)
    return cl


def global_threshold(image, threshold):
    ret, thresh1 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(image, threshold, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()
    print(ret)


def adaptive_threshold(image, blur=False):
    if blur:
        image = cv2.bilateralFilter(image, 17, 35, 35)
    ret, th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    th_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5)
    th_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
    titles = ['Original Image', 'Global Threshold', 'Adaptive Mean Threshold', 'Adaptive Gaussian Threshold']
    images = [image, th, th_mean, th_gaussian]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()


def otsu_binarization(image, blur=False):
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    ret, th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ret_otsu, th_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    titles = ['Original Image', 'Global Threshold', 'Otsu Binarization']
    images = [image, th, th_otsu]

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()
    print("Otsu's threshold is:", ret_otsu)


def erosion(image, iterations):
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    plt.subplot(211), plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(212), plt.imshow(eroded, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


def dilation(image, iterations):
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    plt.subplot(211), plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(212), plt.imshow(dilated, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


img_address = "./test-images/female_35.bmp"
img = cv2.imread(img_address, 0)
equalized, both_images = equalize_histogram(image=img)
adaptive_equalized = CLAHE(image=img)
# double_equalized = CLAHE(image=adaptive_equalized)

# draw_histogram(image=img)
# draw_histogram(image=equalized)
# draw_histogram(image=adaptive_equalized)
# draw_histogram(image=d)


# plt.subplot(121), plt.imshow(adaptive_equalized, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(b, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.show()
