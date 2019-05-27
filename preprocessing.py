import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola, rank)
from skimage.morphology import disk


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
    result = cv2.equalizeHist(image)
    return result


def AHE(image, radius=150):
    result = rank.equalize(image=image, selem=disk(radius))
    return result


def CLAHE(image, clip_limit=2.0, grid_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
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
        images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]

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
        image = cv2.bilateralFilter(src=image, d=17, sigmaColor=35, sigmaSpace=35)
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
        image = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0)

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


def niblack(image, window_size, k, show_result=False, return_result=False):
    binary_global = image > threshold_otsu(image)
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=k)
    binary_niblack = image > thresh_niblack

    if show_result:
        plt.subplot(3, 1, 1), plt.imshow(image, cmap='gray'), plt.title('Original'), plt.axis('off')
        plt.subplot(3, 1, 2), plt.imshow(binary_global, cmap='gray'), plt.title('Global Threshold'), plt.axis('off')
        plt.subplot(3, 1, 3), plt.imshow(binary_niblack, cmap='gray'), plt.title('Niblack Threshold'), plt.axis('off')
        plt.show()

    if return_result:
        return binary_niblack


def sauvola(image, window_size, show_result=False, return_result=False):
    binary_global = image > threshold_otsu(image)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    binary_sauvola = image > thresh_sauvola

    if show_result:
        plt.subplot(3, 1, 1), plt.imshow(image, cmap='gray'), plt.title('Original'), plt.axis('off')
        plt.subplot(3, 1, 2), plt.imshow(binary_global, cmap='gray'), plt.title('Global Threshold'), plt.axis('off')
        plt.subplot(3, 1, 3), plt.imshow(binary_sauvola, cmap='gray'), plt.title('Sauvola Threshold'), plt.axis('off')
        plt.show()

    binary_sauvola = np.array(binary_sauvola, dtype='uint8')
    binary_sauvola = np.where(binary_sauvola == 1, np.uint8(255), np.uint8(0))

    if return_result:
        return binary_sauvola


def canny(image, blur=False, show_result=False, return_result=False):
    if blur:
        image = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0)
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


def erosion(image, kernel_size=8, iterations=2, show_result=False, return_result=False):
    kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)

    if show_result:
        plt.figure(figsize=(10, 10))
        plt.subplot(211), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(eroded, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return eroded


def dilation(image, kernel_size=8, iterations=2, show_result=False, return_result=False):
    kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)

    if show_result:
        plt.figure(figsize=(10, 10))
        plt.subplot(211), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(dilated, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return dilated


def quadtree_decomp(image, min_size, min_std, show_result=False, return_result=False):

    def split_image(image):
        h, w = image.shape[0], image.shape[1]
        off1X = 0
        off1Y = 0
        off2X = 0
        off2Y = 0

        if w >= h:  # split X
            off1X = 0
            off2X = int(w / 2)
            img1 = image[0:h, 0:off2X]
            img2 = image[0:h, off2X:w]
        else:  # split Y
            off1Y = 0
            off2Y = int(h / 2)
            img1 = image[0:off2Y, 0:w]
            img2 = image[off2Y:h, 0:w]

        return off1X, off1Y, img1, off2X, off2Y, img2

    def extract_roi(image, min_std, min_size, offX, offY, roi_list):
        h, w = image.shape[0], image.shape[1]
        m, s = cv2.meanStdDev(image)

        if s >= min_std and max(h, w) > min_size:
            oX1, oY1, im1, oX2, oY2, im2 = split_image(image)

            extract_roi(im1, min_std, min_size, offX + oX1, offY + oY1, roi_list)
            extract_roi(im2, min_std, min_size, offX + oX2, offY + oY2, roi_list)
        else:
            roi_list.append([offX, offY, w, h, m, s])

        return roi_list

    h, w = image.shape[0], image.shape[1]
    input_image = image

    # if not isinstance(input_image, type(None)):
    #     if input_image.ndim > 1:
    #         input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #     else:
    #         pass
    # else:
    #     print('Error on input image: ', input_image)
    #     exit()

    roi_list = extract_roi(image=input_image, min_std=min_std, min_size=min_size, offX=0, offY=0, roi_list=list())
    output_image = input_image.copy()
    for roi in roi_list:
        for element in range(0, len(roi)):
            roi[element] = int(roi[element])
        color = 255  # white color
        if roi[5] < min_std:
            color = 0  # black color

        cv2.rectangle(output_image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), color, 1)

    if show_result:
        if w > h:
            plt.subplot(211)
        else:
            plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])

        if w > h:
            plt.subplot(212)
        else:
            plt.subplot(122)
        plt.imshow(output_image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if return_result:
        return output_image


def imfill(image, threshold, window_size, show_result=False, return_result=False):
    sa = sauvola(image=image, window_size=window_size, show_result=False, return_result=True)
    sa_inv = np.invert(sa)
    sa_inv = np.array(sa_inv, dtype='uint8')

    retval, labels = cv2.connectedComponents(sa_inv)

    hist = np.histogram(labels, retval)[0]
    noise = list()
    hist_copy = np.copy(hist)
    hist_copy = sorted(hist_copy, reverse=True)
    th = hist_copy[threshold]

    for ii in range(len(hist)):
        if hist[ii] > th:
            noise.append(ii)
    gaps = np.ones([len(sa_inv), len(sa_inv[0])], np.bool)

    noise = noise[1:]
    for ii in range(len(sa_inv)):
        # print("---------", ii)
        for num in noise:
            indexes = [i for i, j in enumerate(labels[ii]) if j == num]
            for jj in indexes:
                gaps[ii, jj] = False
                # print(jj)

    if show_result:
        plt.subplot(3, 1, 1), plt.imshow(X=image, cmap='gray'), plt.axis('off')
        plt.subplot(3, 1, 2), plt.imshow(X=sa, cmap='gray'), plt.axis('off')
        plt.subplot(3, 1, 3), plt.imshow(X=gaps, cmap='gray'), plt.axis('off')
        plt.show()

    if return_result:
        return gaps


def eliminate_white_pixels(image):
    image = np.where(image == 255, np.uint8(254), image)
    return image


def eliminate_black_pixels(image):
    image = np.where(image == 0, np.uint8(1), image)
    return image

