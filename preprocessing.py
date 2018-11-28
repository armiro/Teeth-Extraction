import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)


def draw_histogram(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


def match_histogram(source, template):
    """
    Adjust the pixel values of a gray-scale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interpolated_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    hist_matched_img = interpolated_t_values[bin_idx].reshape(oldshape)
    result = np.array(hist_matched_img, dtype='uint8')

    return result


def equalize_histogram(image):
    equ = cv2.equalizeHist(image)
    res = np.hstack((image, equ))
    return equ, res


def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_equalized = clahe.apply(image)
    return adaptive_equalized


def global_threshold(image, threshold, show_result=False, return_result=False):
    ret, thresh1 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(image, threshold, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO_INV)

    if show_result:
        titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()

    if return_result:
        return thresh1


def adaptive_threshold(image, blur=False, show_result=False, return_result=False):
    if blur:
        image = cv2.bilateralFilter(image, 17, 35, 35)
    ret, th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    th_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5)
    th_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)

    if show_result:
        titles = ['Original Image', 'Global Threshold', 'Adaptive Mean Threshold', 'Adaptive Gaussian Threshold']
        images = [image, th, th_mean, th_gaussian]

        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])
        plt.show()

    if return_result:
        return th_mean, th_gaussian


def otsu(image, blur=False, show_result=False, return_result=False):
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    ret, th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ret_otsu, th_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if show_result:
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

    if return_result:
        return th_otsu


def niblack_and_sauvola(image, window_size, show_result=False, return_result=False):
    binary_global = image > threshold_otsu(image)

    window_size = window_size
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=1)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola

    if show_result:
        plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
        plt.title('Original'), plt.axis('off')
        plt.subplot(2, 2, 2), plt.imshow(binary_global, cmap='gray')
        plt.title('Global Threshold'), plt.axis('off')
        plt.subplot(2, 2, 3), plt.imshow(binary_niblack, cmap='gray')
        plt.title('Niblack Threshold'), plt.axis('off')
        plt.subplot(2, 2, 4), plt.imshow(binary_sauvola, cmap='gray')
        plt.title('Sauvola Threshold'), plt.axis('off')

        plt.show()

    if return_result:
        return binary_niblack, binary_sauvola


def canny(image, blur=False, show_result=False, return_result=False):
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(image, 140, 200)

    if show_result:
        plt.subplot(211), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(edges, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return edges


def laplacian(image, show_result=False, return_result=False):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    if show_result:
        plt.subplot(211), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(laplacian, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return laplacian


def erosion(image, kernel_size, iterations, show_result=False, return_result=False):
    kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)

    if show_result:
        plt.subplot(211), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(eroded, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return eroded


def dilation(image, kernel_size, iterations, show_result=False, return_result=False):
    kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)

    if show_result:
        plt.subplot(211), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(dilated, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return dilated


### Load images ###
img_address = "./test-images/T.jpg"
# tmp_address = "./test-images/download.jpg"
img = cv2.imread(img_address, 0)
# tmp = cv2.imread(tmp_address, 0)

### Do the processing tasks ###


### Plot the final result ###
# plt.subplot(211), plt.imshow(img, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.imshow(img, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.show()
