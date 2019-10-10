import cv2
import preprocessing as prep
import matplotlib.pyplot as plt

img_address = './test-auto-cropped/7.bmp'
img = cv2.imread(img_address, 0)
print(img.shape)
# img = prep.CLAHE(image=img)
img = prep.AHE(image=img, radius=150)
plt.imshow(X=img, cmap='gray')
plt.show()
