import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import copy


file_path = "./cropped-images/real_roi_corners.csv"
roi_corners = list()
temp_image_path = "./images/1.bmp"
temp_image = cv2.imread(temp_image_path, 0)
print('base image shape:', temp_image.shape)
height, width = temp_image.shape[:2]

with open(file=file_path, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        roi_corners.append(row)


roi_corners.pop(0)
empty_image = np.zeros(shape=(height, width), dtype=int)
roi_centers = list()
roi_widths = list()
roi_heights = list()
roi_areas = list()

for roi_corner in roi_corners:
    # map the corner points into a list of two numbers [x, y]
    top_left_corner = list(map(int, roi_corner[1].replace("(", "").replace(")", "").split(",")))
    bottom_right_corner = list(map(int, roi_corner[2].replace("(", "").replace(")", "").split(",")))

    # calculate the roi center point
    roi_center_x = int((top_left_corner[0] + bottom_right_corner[0]) / 2.)
    roi_center_y = int((top_left_corner[1] + bottom_right_corner[1]) / 2.)
    roi_center = (roi_center_x, roi_center_y)
    roi_centers.append(roi_center)

    # calculate the roi width, height and area
    roi_width = bottom_right_corner[0] - top_left_corner[0]
    roi_widths.append(roi_width)
    roi_height = bottom_right_corner[1] - top_left_corner[1]
    roi_heights.append(roi_height)
    roi_area = roi_width * roi_height
    roi_areas.append(roi_area)

    # plot this roi_center, also with weighted value (transparency)
    # tmp_image = np.zeros(shape=(height, width), dtype=int)
    cv2.circle(img=empty_image, center=roi_center, radius=2, color=127, thickness=-1)
    # empty_image = cv2.addWeighted(src1=tmp_image, alpha=0.1, src2=empty_image, beta=0.9, gamma=0)

image_center = (int(width/2.), int(height/2.))
cv2.circle(img=empty_image, center=image_center, radius=2, color=255, thickness=-1)
# zoomed_image = empty_image[620:900, 1570:1700]
# plt.imshow(X=zoomed_image, cmap=plt.cm.inferno)
# plt.show()

# calculate the distances from image_center (errors)
distances = list()
for roi_center in roi_centers:
    this_distance = round(number=np.emath.sqrt((roi_center[0] - image_center[0])**2 + (roi_center[1] - image_center[1])**2), ndigits=2)
    distances.append(this_distance)

# calculate the mean-squared of distances (MSE)
mse = 0
for distance in distances:
    mse += (distance ** 2)
mse /= len(distances)

# print the results of errors (centroids, widths, heights and areas)
print('the MSE is:', round(mse, 2))
print('the RMSE is:', round(np.emath.sqrt(mse), 2))
print('the MAE is:', round(np.mean(distances), 2))

print('the average roi width is:', round(np.mean(a=roi_widths), 2))
print('the standard deviation of roi widths is:', round(np.std(a=roi_widths), 2))

print('the average roi height is:', round(np.mean(a=roi_heights), 2))
print('the standard deviation of roi heights is:', round(np.std(a=roi_heights), 2))

print('the average roi area is:', round(np.mean(a=roi_areas), 2))
print('the standard deviation of roi areas is:', round(np.std(a=roi_areas), 2))

