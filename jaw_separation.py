import cv2
import preprocessing as prep
import matplotlib.pyplot as plt
import numpy as np
import time


img = cv2.imread('./test-images/snaked.bmp', 0)
height, width = img.shape[:2]
print('image shape:', img.shape)

upper_jaw = np.zeros(shape=(height, 1))
lower_jaw = np.zeros(shape=(height, 1))
line_points = list()
img = prep.eliminate_black_pixels(image=img)
middle_line_pixels = list()

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

            middle_line_pixels.append(px_idx)
            break
t1 = time.time()
print('elapsed time for jaw separation: %.2f secs' % round(t1-t0, 2))

upper_bound = max(middle_line_pixels)
upper_jaw = upper_jaw[:upper_bound + 1, :]
plt.imshow(X=upper_jaw, cmap='gray')
plt.show()
cv2.imwrite('./test-images/up_jaw.bmp', upper_jaw)

lower_bound = min(middle_line_pixels)
lower_jaw = lower_jaw[lower_bound + 1:, :]
plt.imshow(X=lower_jaw, cmap='gray')
plt.show()
cv2.imwrite('./test-images/low_jaw.bmp', lower_jaw)

