import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
# from sklearn.cluster import KMeans
import preprocessing
# from scipy.ndimage.morphology import binary_fill_holes


def find_poi(image, num_parts):
    """load the image and do the pre-processing section"""
    height, width = image.shape[0], image.shape[1]
    print("image shape is:", image.shape)
    image = preprocessing.CLAHE(image=image)

    """divide image using number of windows (n)"""
    n = num_parts
    window_size = int(width / n)
    parts = list()
    middle_dots = list()
    h_open = int(height / 8.)
    h_close = height - h_open

    for part_idx in range(0, n):
        parts.append(image[h_open:h_close, (window_size * part_idx):(window_size * part_idx + window_size)])

    img_with_dots = copy.deepcopy(x=image)
    ii = -1
    for part in parts:
        ii += 1
        sum_array = list()

        for line_idx in range(0, len(part)):
            line = part[line_idx, :]
            sum_array.append(sum(line))

        # a = width/(n * (max(sum_array) - min(sum_array)))
        # b = -1 * a * min(sum_array)
        # for sum_idx in range(0, len(sum_array)):
        #     sum_array[sum_idx] = (a * sum_array[sum_idx]) + b

        # binarized = list()
        # for sum_element in sum_array:
        #     binarized.append(sum_element > 5)
        # plt.plot(binarized)
        # plt.show()

        tmp_array = copy.deepcopy(x=sum_array)
        # for e in range(0, len(tmp_array)):
        #     tmp_array[e] = (tmp_array[e] > 4000)

        # plt.plot(tmp_array)
        # plt.show()

        minimum = min(sum_array)
        minimum_idx = sum_array.index(minimum) + h_open

        tmp = int(width / n) * ii
        middle_dot = (int((2 * tmp + part.shape[1]) / 2.), minimum_idx)

        # cv2.line(img_with_dots, (tmp, minimum_idx), (tmp + part.shape[1], minimum_idx), 255, 3)
        cv2.circle(img=img_with_dots, center=middle_dot, radius=5, color=(255, 0, 0), thickness=-1)

        middle_dots.append(middle_dot)

    """find the starting point"""
    starting_x = int(width / 2.)
    x_bound = int(width / 5.)
    mid_y = list()
    for point in middle_dots:
        if (point[0] > (starting_x - x_bound)) & (point[0] < (starting_x + x_bound)):
            mid_y.append(point[1])
    starting_y = int(np.mean(a=mid_y))
    supposed_starting_point = (starting_x, starting_y)
    # print(supposed_starting_point)

    distances = list()
    for point in middle_dots:
        if (point[0] > (starting_x - x_bound)) & (point[0] < (starting_x + x_bound)):
            distances.append((point[1] - starting_y) ** 2)
    # print(distances)
    nearest = np.argmin(a=distances)
    # print(nearest)
    poi_idx = int(nearest + int((starting_x - x_bound) / window_size))
    # print(middle_dots[poi])
    poi = middle_dots[poi_idx]
    print(poi)
    return poi


def line_intensity(x1, y1, x2, y2, image):
    empty_img = np.zeros(shape=image.shape)
    cv2.line(empty_img, (x1, y1), (x2, y2), 1, 2)
    next_img = empty_img * image

    # plt.imshow(next_img, cmap='gray')
    # plt.show()

    num = sum(sum(empty_img))
    # print(num)
    mean = sum(sum(next_img)) / num
    maximum = np.max(a=next_img)
    return mean, maximum


def next_poi(image, poi, delta_x, delta_y):
    # cv2.circle(img, sp, 5, 255, -1)

    new_point_x = poi[0] + delta_x
    # cv2.line(img, (new_point_x, sp[1]-20), (new_point_x, sp[1]+20), 255, 1)
    new_y_bottom = poi[1] + delta_y
    new_y_top = poi[1] - delta_y

    def cost_fn(param1, param2):
        fn = 1 * param1 + 0 * param2
        return fn

    all_cost_fn = list()
    for y in range(new_y_top, new_y_bottom, 2):
        # cv2.line(img, sp, (new_point_x, y), 255, 1)

        this_mean, this_max = line_intensity(x1=poi[0], y1=poi[1], x2=new_point_x, y2=y, image=img)
        all_cost_fn.append(cost_fn(param1=this_mean, param2=this_max))

    min_cost_idx = np.argmin(all_cost_fn)
    # print(min_cost_idx)

    new_y = new_y_top + (2 * min_cost_idx)
    # cv2.line(image, poi, (new_point_x, new_y), 255, 5)

    # plt.imshow(image, cmap='gray')
    # plt.show()

    return new_point_x, new_y


def draw_mid_line(image, points):
    for point_idx in range(len(points) - 1):
        cv2.line(image, (points[point_idx][0], points[point_idx][1]), (points[point_idx + 1][0], points[point_idx + 1][1]), 255, 3)
        cv2.circle(image, (points[point_idx][0], points[point_idx][1]), 10, 255, -1)
    plt.imshow(image, cmap='gray')
    plt.show()


img = cv2.imread('./test-auto-cropped/4.bmp', 0)
height, width = img.shape[0], img.shape[1]
# mn, mx = line_intensity(x1=37, y1=73, x2=48, y2=79, image=img)
# print(mn, mx)
sp = find_poi(image=img, num_parts=20)
# print(sp)
delta_x = int(width / 20)
delta_y = 50
flag = True
# img = preprocessing.CLAHE(image=img)

img = preprocessing.imfill(image=img, threshold=10, return_result=1)
img = np.array(img, dtype='uint8')
img = cv2.blur(img, (40, 40))

plt.imshow(img, cmap='gray')
plt.show()

points = list([sp])
while flag:
    next_point = next_poi(image=img, poi=points[-1], delta_x=delta_x, delta_y=delta_y)
    if next_point[0] > width:
        flag = False
        last_point = next_poi(image=img, poi=points[-1], delta_x=(width - points[-1][0]), delta_y=delta_y)
        points.append(last_point)
    else:
        points.append(next_point)

flag = True
points.append(sp)

while flag:
    next_point = next_poi(image=img, poi=points[-1], delta_x=-delta_x, delta_y=delta_y)
    if next_point[0] < 0:
        flag = False
        last_point = next_poi(image=img, poi=points[-1], delta_x=(-points[-1][0]), delta_y=delta_y)
        points.append(last_point)
    else:
        points.append(next_point)


points = np.unique(ar=points, axis=0)
points.sort(axis=0, kind='quicksort')
draw_mid_line(image=img, points=points)

