import cv2
import preprocessing
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import copy
import clustering

img_address = './test-revised/5.bmp'
img = cv2.imread(img_address, 0)
height, width = img.shape[0], img.shape[1]

img = preprocessing.CLAHE(image=img, clip_limit=2., grid_size=8)


# img = cv2.bilateralFilter(img, 17, 35, 35)
# img = cv2.blur(img, (15, 15))
# img = cv2.GaussianBlur(img, (5, 5), 0)
# img = cv2.medianBlur(img, 15)

img = preprocessing.dilation(image=img, kernel_size=5, iterations=2, return_result=1)

sum_array = list()
for line_idx in range(width):
    line = img[:int(height/2.), line_idx]
    sum_array.append(sum(line))

mean = np.mean(a=sum_array)
ln = mlines.Line2D(xdata=[0, width], ydata=[mean, mean], linewidth=2)
ax = plt.gca()
ax.add_line(ln)
plt.plot(sum_array)
# plt.ylim((height, 0))
plt.show()
plt.imshow(X=img, cmap='gray')
plt.show()

