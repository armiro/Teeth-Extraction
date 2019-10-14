# import csv
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
    elif line is not '\n':
        line = line[:-2]
        points = line.split(sep=';')
    else:
        pts.append((num, points))

# images = os.scandir(path='./lower_jaws/')

for image_name in glob.glob(pathname="./lower_jaws/**.bmp"):
    print(image_name)
    img_num = image_name.split('\\')[-1].split('_')[0]
    print("image number is:", img_num)
    img = cv2.imread(image_name, 0)
    height, width = img.shape[:2]
    upsize_coef = round((width / 216), ndigits=2)

    for pt in pts:
        if pt[0] == img_num:
            print("point found:", pt[0])
            coordinates = [int(int(point) * upsize_coef) for point in pt[1]]
            org = coordinates[30:60]
            top_dev = coordinates[:30]
            bottom_dev = coordinates[60:]

            org = np.array(org)
            top_dev = np.array(top_dev)
            bottom_dev = np.array(bottom_dev)

            top_dev = list(org + top_dev)
            bottom_dev = list(org + bottom_dev)
            org = list(org)

            org_tmp = org[:]
            top_dev_tmp = top_dev[:]
            bottom_dev_tmp = bottom_dev[:]
            for i in range(len(org)):
                try:
                    if org[i] == 0:
                        org_tmp.remove(org[i])
                        bottom_dev_tmp.remove(bottom_dev[i])
                        top_dev_tmp.remove(top_dev[i])
                except:
                    pass
            org = org_tmp[:]
            top_dev = top_dev_tmp[:]
            bottom_dev = bottom_dev_tmp[:]

            bottom_dev = [0 if i < 0 else i for i in bottom_dev]
            top_dev = [0 if i < 0 else i for i in top_dev]

            org = [0] + org + [width]
            bottom_dev = [0] + bottom_dev + [width]
            top_dev = [0] + top_dev + [width]

            print(top_dev)
            print(org)
            print(bottom_dev)

            white = (255, 255, 255)

            for idx in range(0, len(org)-1):
                tooth_corners = np.array([[(top_dev[idx], 0), (top_dev[idx + 1], 0),
                                           (org[idx + 1], int(height / 2)), (bottom_dev[idx + 1], height),
                                           (bottom_dev[idx], height), (org[idx], int(height / 2))]],
                                         dtype=np.int32)
                # print(tooth_corners)
                tooth_mask = np.zeros(img.shape, dtype=np.uint8)
                cv2.fillPoly(tooth_mask, tooth_corners, white)
                img_copy = copy.deepcopy(x=img)
                tooth_img = cv2.bitwise_and(img_copy, tooth_mask)

                left_bound = min([top_dev[idx], org[idx], bottom_dev[idx]])
                right_bound = max([top_dev[idx + 1], org[idx + 1], bottom_dev[idx + 1]])
                tooth_img = tooth_img[:, left_bound:right_bound]

                # create path for each jaw (skip, if the path exists)
                os.makedirs('./extracted-images/%d' % int(img_num), exist_ok=True)

                # save each tooth with a name in the newly created path
                cv2.imwrite('./extracted-images/%d/L%d.bmp' % (int(img_num), idx + 1), tooth_img)

                plt.imshow(X=tooth_img, cmap='gray')
                plt.show()
            break

# img_name = img_path[img_path.rfind('/')+1:img_path.rfind('.bmp')]
# img_num, jaw_type = img_name.split("_")
# csv_file = open(file=csv_path, mode='r', newline='')
# coordinates = list(csv.reader(csv_file, delimiter=',', quotechar='"'))


# every element in every "row" of the "coordinates", is saved as string. we need to map it into integer,
# and also multiply it by the "upsize_coef" at the same time
# coordinates = [list(map(lambda x: int(int(x) * upsize_coef), row)) for row in coordinates]

# top_dev = [0] + coordinates[0] + [width]
# org = [0] + coordinates[1] + [width]
# bottom_dev = [0] + coordinates[2] + [width]

# white = (255, 255, 255)
# num_lines = len(coordinates[0])
#
# for idx in range(0, num_lines):
#     tooth_corners = np.array([[(top_dev[idx], 0), (top_dev[idx + 1], 0),
#                                (org[idx + 1], int(height / 2)), (bottom_dev[idx + 1], height),
#                                (bottom_dev[idx], height), (org[idx], int(height / 2))]],
#                              dtype=np.int32)
#     # print(tooth_corners)
#     tooth_mask = np.zeros(img.shape, dtype=np.uint8)
#     cv2.fillPoly(tooth_mask, tooth_corners, white)
#     img_copy = copy.deepcopy(x=img)
#     tooth_img = cv2.bitwise_and(img_copy, tooth_mask)
#
#     left_bound = min([top_dev[idx], org[idx], bottom_dev[idx]])
#     right_bound = max([top_dev[idx + 1], org[idx + 1], bottom_dev[idx + 1]])
#     tooth_img = tooth_img[:, left_bound:right_bound]
#
#     # create path for each jaw (skip, if the path exists)
#     os.makedirs('./extracted-images/%d' % int(img_num), exist_ok=True)
#
#     # save each tooth with a name in the newly created path
#     cv2.imwrite('./extracted-images/%d/%s%d.bmp' % (int(img_num), jaw_type[0].upper(), idx+1), tooth_img)
#     plt.imshow(X=tooth_img, cmap='gray')
#     plt.show()
