import cv2
import preprocessing
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import copy
import clustering

img_address = './test-revised/9.bmp'
img = cv2.imread(img_address, 0)
height, width = img.shape[0], img.shape[1]
tmp = copy.deepcopy(x=img)

img = preprocessing.CLAHE(image=img, clip_limit=2., grid_size=8)
# img, res = preprocessing.equalize_histogram(image=img)

# img = cv2.bilateralFilter(img, 17, 35, 35)
# img = cv2.blur(img, (15, 15))
# img = cv2.GaussianBlur(img, (5, 5), 0)
# img = cv2.medianBlur(img, 15)

# img = preprocessing.erosion(image=img, kernel_size=15, iterations=1, return_result=1)
img = preprocessing.dilation(image=img, kernel_size=5, iterations=2, return_result=1)

sum_array = list()
for line_idx in range(width):
    line = img[:int(height/2.), line_idx]
    sum_array.append(sum(line))

mean = int(np.mean(a=sum_array))
ln = mlines.Line2D(xdata=[0, width], ydata=[mean, mean], linewidth=2)
ax = plt.gca()
ax.add_line(ln)
plt.plot(sum_array)
# plt.ylim((height, 0))
plt.show()

for sum_element in range(0, len(sum_array)-2):
    if (sum_array[sum_element] <= mean) & (sum_array[sum_element+1] >= mean):
        left_eoi = sum_element
        break
cv2.line(tmp, (left_eoi, 0), (left_eoi, height), 0, 3)

for sum_element in range(len(sum_array)-2, 0, -1):
    if (sum_array[sum_element+1] <= mean) & (sum_array[sum_element] >= mean):
        right_eoi = sum_element
        break
cv2.line(tmp, (right_eoi, 0), (right_eoi, height), 0, 3)

plt.imshow(X=tmp, cmap='gray')
plt.show()

plt.imshow(X=img, cmap='gray')
plt.show()

