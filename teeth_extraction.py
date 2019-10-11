import csv
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import glob


lower_text_path = './lower_jaws/low.txt'
lower_text_file = open(file=lower_text_path, mode='r')
lines = lower_text_file.readlines()

pts = list()
for line in lines:
    if line.find('.bmp') is not -1:
        num = line.split(sep='_')[0]
    elif line is not '/n':
        line = line[:-2]
        points = line.split(sep=';')
        top_dev = points[:30]
        org = points[30:60]
        bottom_dev = points[60:]
    else:
        pts.append((num, top_dev, org, bottom_dev))

images = os.scandir(path='./lower_jaws')

for image_name in glob.glob(pathname="**.bmp"):
    img_num = image_name.split("_")[0]
    img = cv2.imread(image_name, 0)
    height, width = img.shape[:2]
    upsize_coef = round((width / 216), ndigits=2)

    for pt in pts:
        if pt[0] is img_num:


    coordinates = [int(int(x) * upsize_coef) for point in points]


# img_name = img_path[img_path.rfind('/')+1:img_path.rfind('.bmp')]
# img_num, jaw_type = img_name.split("_")
# csv_file = open(file=csv_path, mode='r', newline='')
# coordinates = list(csv.reader(csv_file, delimiter=',', quotechar='"'))


# every element in every "row" of the "coordinates", is saved as string. we need to map it into integer,
# and also multiply it by the "upsize_coef" at the same time
# coordinates = [list(map(lambda x: int(int(x) * upsize_coef), row)) for row in coordinates]

top_coordinates = [0] + coordinates[0] + [width]
middle_coordinates = [0] + coordinates[1] + [width]
bottom_coordinates = [0] + coordinates[2] + [width]

white = (255, 255, 255)
num_lines = len(coordinates[0])

for idx in range(0, num_lines):
    tooth_corners = np.array([[(top_coordinates[idx], 0), (top_coordinates[idx+1], 0),
                               (middle_coordinates[idx+1], int(height/2)), (bottom_coordinates[idx+1], height),
                               (bottom_coordinates[idx], height), (middle_coordinates[idx], int(height/2))]],
                             dtype=np.int32)
    # print(tooth_corners)
    tooth_mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillPoly(tooth_mask, tooth_corners, white)
    img_copy = copy.deepcopy(x=img)
    tooth_img = cv2.bitwise_and(img_copy, tooth_mask)

    left_bound = min([top_coordinates[idx], middle_coordinates[idx], bottom_coordinates[idx]])
    right_bound = max([top_coordinates[idx + 1], middle_coordinates[idx + 1], bottom_coordinates[idx + 1]])
    tooth_img = tooth_img[:, left_bound:right_bound]

    # create path for each jaw (skip, if the path exists)
    os.makedirs('./extracted-images/%d' % int(img_num), exist_ok=True)
    
    # save each tooth with a name in the newly created path
    cv2.imwrite('./extracted-images/%d/%s%d.bmp' % (int(img_num), jaw_type[0].upper(), idx+1), tooth_img)
    plt.imshow(X=tooth_img, cmap='gray')
    plt.show()

