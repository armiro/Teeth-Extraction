import cv2
import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from ROI_revision import revise_vertical_boundaries

img_address = './test-revised/1.bmp'
img = cv2.imread(img_address, 0)
revise_vertical_boundaries(image=img, show_result=True, return_result=False)
