import numpy as np
import cv2
import matplotlib.pyplot as plt


def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    cl = clahe.apply(image)
    return cl


img_address = "./test-images/source.jpg"
img = cv2.imread(img_address, 0)
adaptive_equalized = CLAHE(image=img)

matrix = [[-2, 0, 2],
          [-2, 0, 2],
          [-2, 0, 2]]
kernel = np.array(matrix, dtype=np.float32)/1.0
filtered = cv2.filter2D(adaptive_equalized, -1, kernel)

left_boundary = int(len(filtered[0]) / 4.0)
total_intensity = 0
all_intensities = list()

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
    all_intensities.append(float(this_intensity))

mean_intensity = float(mean_intensity)*2
all_intensities.extend([0.0, 0.0])
for idx in range(0, len(all_intensities)-2):
    if (all_intensities[idx] > mean_intensity) & (all_intensities[idx+1] > mean_intensity) & (all_intensities[idx+2] > mean_intensity):
        adaptive_equalized = cv2.line(adaptive_equalized, (idx+1, 0), (idx+1, 390), 255, 2)
        break


plt.subplot(211), plt.imshow(adaptive_equalized, cmap='gray')
# plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(filtered, cmap='gray')
# plt.title('CLAHE')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(filtered, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
