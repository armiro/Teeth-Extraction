import cv2
import preprocessing

img_address = './test-cropped-images/1.bmp'
img = cv2.imread(img_address, 0)
height, width = img.shape[0], img.shape[1]
adp_equ = preprocessing.CLAHE(image=img)

