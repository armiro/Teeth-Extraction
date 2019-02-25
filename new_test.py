import cv2
# import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import time


img = cv2.imread('./test.bmp', 0)
height, width = img.shape[:2]
print('image shape:', img.shape)
upper_jaw = np.zeros(shape=(height, 1))
lower_jaw = np.zeros(shape=(height, 1))
line_points = list()

t0 = time.time()
for column in range(0, width):
    this_column = img[:, column]
    for px_idx in range(0, len(this_column)):
        if this_column[px_idx] == 255:
            line_points.append([column, px_idx])
            up = np.vstack(tup=(img[:px_idx, column:column+1], np.zeros(shape=(height-px_idx, 1))))
            upper_jaw = np.hstack(tup=(upper_jaw, up))

            dn = np.vstack(tup=(np.zeros(shape=(px_idx+1, 1)), img[px_idx+1:, column:column+1]))
            lower_jaw = np.hstack(tup=(lower_jaw, dn))
            break
t1 = time.time()
plt.imshow(X=upper_jaw, cmap='gray')
plt.show()
plt.imshow(X=lower_jaw, cmap='gray')
plt.show()
print('elapsed time for jaw separation: %.2f secs' % round(t1-t0, 2))

shifted_line_points = list()
shifted_line_points2 = list()
for line_point_idx in range(0, len(line_points)):
    this_point = line_points[line_point_idx]
    this_point[1] -= 100
    shifted_line_points.append(this_point)

for line_point_idx in range(0, len(line_points)):
    this_point = line_points[line_point_idx]
    this_point[1] -= 50
    shifted_line_points.append(this_point)


line_intensity = list()
for shifted_line_point in shifted_line_points:
    this_intensity = img[shifted_line_point[1], shifted_line_point[0]]
    line_intensity.append(this_intensity)

line_intensity2 = list()
for shifted_line_point in shifted_line_points2:
    this_intensity = img[shifted_line_point[1], shifted_line_point[0]]
    line_intensity2.append(this_intensity)


plt.plot(line_intensity)
plt.show()

plt.plot(line_intensity2)
plt.show()

# line_intensity = np.array(line_intensity)
# min_indices = line_intensity.argsort()[:16]
# print(min_indices)

