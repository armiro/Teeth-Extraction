import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import copy


def find_middle_points(image, num_parts, show_result=False, return_result=False):
    n = num_parts
    height, width = image.shape[0], image.shape[1]
    window_size = int(width / n) + 1
    parts = list()
    middle_points = list()
    h_open = int(height / 8.)
    h_close = height - h_open

    # divide image into "n" parts
    for part_idx in range(0, n):
        parts.append(image[h_open:h_close, (window_size * part_idx):(window_size * part_idx + window_size)])

    # copy the original image to plot the middle points on it
    image_with_dots = copy.deepcopy(x=image)

    # for each part, calculate the middle dot
    for part_idx in range(0, n):
        this_part = parts[part_idx]
        row_intensities = list()
        part_height, part_width = this_part.shape[0], this_part.shape[1]

        # for each row in a part, calculate the sum of all its pixels' intensities
        for row_idx in range(0, part_height):
            this_row = this_part[row_idx, :]
            row_intensities.append(sum(this_row))

        # find the row with the minimum intensity
        min_row_intensities_idx = np.argmin(a=row_intensities) + h_open

        # the middle dot has the x=middle of this part, y=index of the row with minimum intensity
        # tmp = int(width / n) * part_idx
        middle_of_this_part = int((part_idx + 0.5) * part_width)
        this_middle_point = (middle_of_this_part, min_row_intensities_idx)
        middle_points.append(this_middle_point)

        # plot this middle point on the copied image
        cv2.circle(img=image_with_dots, center=this_middle_point, radius=5, color=(255, 0, 0), thickness=-1)

    if show_result:
        plt.subplot(2, 1, 1), plt.imshow(X=image, cmap='gray')
        plt.subplot(2, 1, 2), plt.imshow(X=image_with_dots, cmap='gray')
        plt.show()

    if return_result:
        return middle_points


def fit_graph(image, middle_points):
    height, width = image.shape[0], image.shape[1]
    x = list()
    y = list()
    y_tmp = list()

    for point in middle_points:
        x.append(point[0])
        y.append(point[1])
        y_tmp.append((point[1]))

    z = np.polyfit(x=x, y=y_tmp, deg=2, rcond=None, full=False)
    p = np.poly1d(z)

    xp = np.linspace(start=0, stop=width, num=width)
    plt.plot(x, y_tmp, '.', xp, p(xp), '-')
    plt.ylim(height, 0), plt.xlim(0, width)
    plt.imshow(image, cmap='gray')
    plt.show()


def find_starting_point(image, middle_points, show_result=False, return_result=False):
    height, width = image.shape[0], image.shape[1]
    x_supposed_sp = int(width / 2.)
    y_supposed_sp = int(height / 2.)
    x_bound = int(width / 5.)
    y_bound = int(height / 6.)
    mid_y = list()

    # find points which are horizontally located in the middle of image
    for point in middle_points:
        if (point[0] > (x_supposed_sp - x_bound)) & (point[0] < (x_supposed_sp + x_bound)):
            if (point[1] > (y_supposed_sp - y_bound)) & (point[1] < (y_supposed_sp + y_bound)):
                mid_y.append(point[1])

    # not letting mid_y to be empty if there were no acceptable points to add
    if not mid_y:
        mid_y.append(y_supposed_sp)

    # define the estimated starting point (supposed_sp)
    y_supposed_sp = int(np.mean(a=mid_y))
    supposed_sp = (x_supposed_sp, y_supposed_sp)

    # calculate the vertical distances between each horizontally located middle point
    # and the supposed starting point
    distances_from_supposed_sp = list()
    for point in middle_points:
        if (point[0] > (x_supposed_sp - x_bound)) & (point[0] < (x_supposed_sp + x_bound)):
            distances_from_supposed_sp.append((point[1] - y_supposed_sp) ** 2)
    # print("distances from the supposed sp are:", distances_from_supposed_sp)

    # find the nearest middle point to the supposed starting point
    nearest = np.argmin(a=distances_from_supposed_sp)

    # find the window_size based on the length of middle_points
    window_size = int(width / len(middle_points)) + 1
    sp_idx = int(nearest + int((x_supposed_sp - x_bound) / window_size))
    sp = middle_points[sp_idx]

    # plot the sp on the image
    image_with_sp = copy.deepcopy(x=image)
    cv2.circle(img=image_with_sp, center=sp, radius=10, color=(255, 0, 0), thickness=-1)

    if show_result:
        print("starting point is located in:", sp)
        plt.imshow(X=image_with_sp, cmap='gray')
        plt.show()

    if return_result:
        return sp


def calculate_distance(this_point, that_point):
    dist = math.sqrt((this_point[0] - that_point[0]) ** 2 + (this_point[1] - that_point[1]) ** 2)
    return round(dist, 3)


def remove_outliers_by_distance(image, middle_points, starting_point, show_result=False, return_result=False):
    sp_idx = middle_points.index(starting_point)
    height, width = image.shape[0], image.shape[1]
    window_size = int(width / len(middle_points)) + 1
    middle_points_optimized = list()

    # crawl through the right of the starting point
    right_idx_vector = list([sp_idx])
    for idx in range(sp_idx + 1, len(middle_points)):

        # calculate the distance between each middle point and the next middle point
        distance = calculate_distance(this_point=middle_points[idx], that_point=middle_points[right_idx_vector[-1]])

        # define a threshold as the removal condition.
        # as we get farther than the point, we need to modify the threshold
        threshold = math.sqrt((1 + (idx - right_idx_vector[-1]) ** 2))
        threshold = threshold * window_size
        if (distance ** 0.9) <= threshold:
            right_idx_vector.append(idx)

    for idx in right_idx_vector:
        middle_points_optimized.append(middle_points[idx])

    # crawl through the left of the starting point and do the same thing like above
    left_idx_vector = list([sp_idx])
    for idx in range(sp_idx - 1, 0, -1):
        distance = calculate_distance(this_point=middle_points[idx], that_point=middle_points[left_idx_vector[-1]])
        threshold = math.sqrt((1 + (idx - left_idx_vector[-1]) ** 2))
        threshold = threshold * window_size
        if (distance ** 0.9) <= threshold:
            left_idx_vector.append(idx)

    for idx in left_idx_vector:
        middle_points_optimized.append(middle_points[idx])

    # now that we have the optimized middle points, sort them based on the 'x' values
    middle_points_optimized = sorted(middle_points_optimized, key=lambda tup: tup[0])

    # calculate the first and the last points
    first_point = middle_points_optimized[0]
    last_point = middle_points_optimized[-1]

    # add to points with the same 'y' as the first and the last points and add them
    before_first_point = (0, first_point[1])
    after_last_point = (width, last_point[1])
    middle_points_optimized.insert(0, before_first_point)
    middle_points_optimized.append(after_last_point)

    # draw the optimized middle points on the image
    image_with_optm_points = copy.deepcopy(x=image)
    for idx in range(0, len(middle_points_optimized)):
        cv2.circle(img=image_with_optm_points, center=middle_points_optimized[idx], radius=10, color=255, thickness=-1)

    # connect the points together to draw the middle line of the jaw
    image_with_line = copy.deepcopy(x=image)
    for idx in range(0, len(middle_points_optimized)-1):
        cv2.line(image_with_line, middle_points_optimized[idx], middle_points_optimized[idx + 1], 255, 1)

    # draw the raw middle points on another image in order to show as the result
    image_with_points = copy.deepcopy(x=image)
    for idx in range(0, len(middle_points)):
        cv2.circle(img=image_with_points, center=middle_points[idx], radius=10, color=255, thickness=-1)

    if show_result:
        plt.subplot(3, 1, 1), plt.imshow(X=image_with_points, cmap='gray'), plt.axis('off')
        plt.subplot(3, 1, 2), plt.imshow(X=image_with_optm_points, cmap='gray'), plt.axis('off')
        plt.subplot(3, 1, 3), plt.imshow(X=image_with_line, cmap='gray'), plt.axis('off')
        plt.show()

    if return_result:
        return middle_points_optimized, image_with_line


def draw_middle_line(image, num_parts):
    mid_points = find_middle_points(image=image, num_parts=num_parts, return_result=True)
    sp = find_starting_point(image=image, middle_points=mid_points, return_result=True)
    _, final_img = remove_outliers_by_distance(image=image, middle_points=mid_points, starting_point=sp,
                                               return_result=True)
    return final_img

