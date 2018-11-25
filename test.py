import cv2
import matplotlib.pyplot as plt
import numpy as np


def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    cl = clahe.apply(image)
    return cl


img_address = "./test-images/female_27.bmp"
img = cv2.imread(img_address, 0)
# img = img[120:330, 120:620]
adaptive_equalized = CLAHE(image=img)
matrix = [[-2, 0, 2],
          [-2, 0, 2],
          [-2, 0, 2]]
kernel = np.array(matrix, dtype=np.float32)/1.0
filtered = cv2.filter2D(adaptive_equalized, -1, kernel)

left_boundary = int(len(filtered[0]) / 4.0)
total_intensity = 0

for i in range(0, left_boundary):
    this_row = 0
    for g in filtered:
        this_row += g[i]
    total_intensity += round(this_row/len(filtered[0]), 2)

mean_intensity = round(total_intensity / left_boundary, 2)
print("average intensity is:", mean_intensity)

for i in range(0, left_boundary):
    this_row = 0
    for g in filtered:
        this_row += g[i]
    this_intensity = round(this_row/len(filtered[0]), 2)
    print("accumulated intensity of column %d is %0.2f" % (i, this_intensity))


adaptive_equalized = cv2.line(adaptive_equalized, (63, 0), (63, 390), 255, 2)

plt.subplot(211), plt.imshow(adaptive_equalized, cmap='gray')
# plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(filtered, cmap='gray')
# plt.title('CLAHE')
plt.xticks([]), plt.yticks([])
plt.show()

# plt.imshow(filtered, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.show()

