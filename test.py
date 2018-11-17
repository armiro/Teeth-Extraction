import cv2
import matplotlib.pyplot as plt
import numpy as np


def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(image)
    return cl


img_address = "./test-images/female_35.bmp"
img = cv2.imread(img_address, 0)
adaptive_equalized = CLAHE(image=img)

