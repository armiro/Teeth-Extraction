import numpy as np
import cv2
import preprocessing
from middle_line_via_points import find_middle_points, find_starting_point


def calculate_line_intensity_features(this_point, that_point, image):
    # create a black image and draw the line with a white color
    empty_image = np.zeros(shape=image.shape)
    cv2.line(empty_image, this_point, that_point, 1, 2)

    # keep the image's pixels which are located inside the plotted line
    remained_pixels = empty_image * image

    # calculate the number of pixels located inside the plotted line
    num_remained_pixels = sum(sum(empty_image))
    # print(num)
    # calculate the mean and the maximum of remained-pixels' intensities
    mean_intensities = sum(sum(remained_pixels)) / num_remained_pixels
    max_intensity = np.max(a=remained_pixels)
    return mean_intensities, max_intensity


def find_next_point(image, poi, delta_x, delta_y, v_stride):
    new_x = poi[0] + delta_x
    new_y_bottom = poi[1] + delta_y
    new_y_top = poi[1] - delta_y

    all_cost_fn = list()
    for y in range(new_y_top, new_y_bottom, v_stride):
        this_mean, this_max = calculate_line_intensity_features(this_point=poi, that_point=(new_x, y), image=image)
        cost_fn = (1 * this_mean) + (0 * this_max)
        all_cost_fn.append(cost_fn)

    min_cost_idx = np.argmin(all_cost_fn)
    new_y = new_y_top + (2 * min_cost_idx)
    new_point = (new_x, new_y)

    return new_point


def find_points(image, num_parts, v_bound, v_stride):
    # find the starting point
    sp = find_starting_point(image=image,
                             middle_points=find_middle_points(image=image, num_parts=num_parts, return_result=True),
                             return_result=True)
    image_width = image.shape[1]
    delta_x = int(image_width / num_parts)
    delta_y = v_bound
    flag = True

    prep_image = preprocessing.imfill(image=image, window_size=125, threshold=10, return_result=1)
    prep_image = np.array(prep_image, dtype='uint8')
    prep_image = cv2.blur(prep_image, (50, 50))

    points = list([sp])
    while flag:
        next_point = find_next_point(image=prep_image, poi=points[-1], delta_x=delta_x,
                                     delta_y=delta_y, v_stride=v_stride)
        if next_point[0] > image_width:
            flag = False
            last_point = find_next_point(image=prep_image, poi=points[-1], delta_x=(image_width - points[-1][0]),
                                         delta_y=delta_y, v_stride=v_stride)
            points.append(last_point)
        else:
            points.append(next_point)

    flag = True
    points.append(sp)

    while flag:
        next_point = find_next_point(image=prep_image, poi=points[-1], delta_x=-delta_x,
                                     delta_y=delta_y, v_stride=v_stride)
        if next_point[0] < 0:
            flag = False
            last_point = find_next_point(image=prep_image, poi=points[-1], delta_x=(-points[-1][0]),
                                         delta_y=delta_y, v_stride=v_stride)
            points.append(last_point)
        else:
            points.append(next_point)

    points = np.unique(ar=points, axis=0)
    points = sorted(points, key=lambda tup: tup[0])
    return points


def draw_middle_line(image, points):
    for point_idx in range(len(points) - 1):
        cv2.line(image, (points[point_idx][0], points[point_idx][1]),
                 (points[point_idx + 1][0], points[point_idx + 1][1]), 255, 1)
    return image

