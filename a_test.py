import cv2
import preprocessing
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
# import copy
# import clustering

img_address = './test-revised/3.bmp'
img = cv2.imread(img_address, 0)

img = preprocessing.CLAHE(image=img, clip_limit=2., grid_size=8)
height, width = img.shape[0], img.shape[1]
print(img.shape)
img_resized = cv2.resize(img, (0, 0), fx=(500/width), fy=(200/height))

resized_height, resized_width = img_resized.shape[0], img_resized.shape[1]
linewidth = int(width/400.)

# img = cv2.bilateralFilter(img, 17, 35, 35)
# img = cv2.blur(img, (15, 15))
# img = cv2.GaussianBlur(img, (5, 5), 0)
# img = cv2.medianBlur(img, 15)

"""vertical edges filter: left boundary"""
vertical_edge_detector = np.zeros(shape=(3, 3))
for ii in range(0, len(vertical_edge_detector)):
    vertical_edge_detector[ii, 0] = -1
    vertical_edge_detector[ii, -1] = 1

print(vertical_edge_detector)

v_kernel = np.array(vertical_edge_detector, dtype=np.float32) / 1.0
left_edge = cv2.filter2D(img_resized, -1, v_kernel)
"""end"""

left_edge = preprocessing.dilation(image=left_edge, kernel_size=5, iterations=3, return_result=1)
# img = cv2.blur(img, (15, 15))

sum_array = list()
for line_idx in range(resized_width):
    line = left_edge[:, line_idx]
    sum_array.append(sum(line))

mean = int(np.mean(a=sum_array))
ln = mlines.Line2D(xdata=[0, resized_width], ydata=[mean, mean], linewidth=2)
ax = plt.gca()
ax.add_line(ln)
plt.plot(sum_array)
plt.show()

for sum_element in range(0, len(sum_array) - 2):
    if (sum_array[sum_element] <= mean) & (sum_array[sum_element+1] >= mean):
        left_eoi = int(sum_element * width / 500)
        break
cv2.line(img, (left_eoi, 0), (left_eoi, height), 0, linewidth)
# cv2.line(img_resized, (sum_element, 0), (sum_element, resized_height), 0, linewidth)
print("left eoi is:", left_eoi)


"""vertical edges filter: right boundary"""
vertical_edge_detector = np.zeros(shape=(3, 3))
for ii in range(0, len(vertical_edge_detector)):
    vertical_edge_detector[ii, 0] = 1
    vertical_edge_detector[ii, -1] = -1

print(vertical_edge_detector)

v_kernel = np.array(vertical_edge_detector, dtype=np.float32) / 1.0
right_edge = cv2.filter2D(img_resized, -1, v_kernel)
"""end"""

right_edge = preprocessing.dilation(image=right_edge, kernel_size=5, iterations=3, return_result=1)
# img = cv2.blur(img, (15, 15))

sum_array = list()
for line_idx in range(resized_width):
    line = right_edge[:, line_idx]
    sum_array.append(sum(line))


for sum_element in range(len(sum_array) - 2, 0, -1):
    if (sum_array[sum_element+1] <= mean) & (sum_array[sum_element] >= mean):
        right_eoi = int(sum_element * width / 500)
        break
cv2.line(img, (right_eoi, 0), (right_eoi, height), 0, linewidth)
# cv2.line(img_resized, (sum_element, 0), (sum_element, resized_height), 0, linewidth)
print("right eoi is:", right_eoi)

plt.imshow(X=img_resized, cmap='gray')
plt.show()

plt.imshow(X=img, cmap='gray')
plt.show()

