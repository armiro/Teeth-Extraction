import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import glob

lower_text_path = './lower_jaws/low.txt'
lower_text_file = open(file=lower_text_path, mode='r')
lines = lower_text_file.readlines()

lower_rev_path = './lower_jaws/low_revision.txt'
lower_rev_file = open(file=lower_rev_path, mode='r')
rev_indices = lower_rev_file.readlines()

pts = list()
for line in lines:
    if line.find('.bmp') is not -1:
        num = line.split(sep='_')[0]
    elif line is not '\n':
        line = line[:-2]
        points = line.split(sep=';')
    else:
        pts.append((num, points))

rev_pts = list()
for line in rev_indices:
    if line.find('.bmp') is not -1:
        num = line.split(sep='_')[0]
    elif line is not '\n':
        line = line[:-2]
        points = line.split(sep=';')
        rev_pts.append((num, points))

# images = os.scandir(path='./lower_jaws/')

for image_name in glob.glob(pathname="./lower_jaws/**.bmp"):
    img_num = image_name.split('\\')[-1].split('_')[0]
    print("image number is:", img_num)
    img = cv2.imread(image_name, 0)
    height, width = img.shape[:2]
    mid = int(height / 2.)
    upsize_coef = round((width / 216), ndigits=2)
    print('up-size coefficient=', upsize_coef)
    rev = [e[1] for e in rev_pts if e[0] == img_num][0]
    rev = [int(e) for e in rev]

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

            bottom_dev = [0 if i < 0 else i for i in bottom_dev]
            top_dev = [0 if i < 0 else i for i in top_dev]

            # 'rev' file is brought using MATLAB, hence the indices start from 1,
            # whereas in python indices start from 0
            org = [org[i - 1] for i in rev]
            top_dev = [top_dev[i - 1] for i in rev]
            bottom_dev = [bottom_dev[i - 1] for i in rev]

            org = [0] + org + [width]
            bottom_dev = [0] + bottom_dev + [width]
            top_dev = [0] + top_dev + [width]

            print('top:', top_dev)
            print('org:', org)
            print('bottom:', bottom_dev)
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
                # os.makedirs('./extracted-images/%d' % int(img_num), exist_ok=True)

                # save each tooth with a name in the newly created path
                # cv2.imwrite('./extracted-images/%d/L%d.bmp' % (int(img_num), idx + 1), tooth_img)

                # plt.imshow(X=tooth_img, cmap='gray')
                # plt.show()

            # draw the lines on the initial image and save it
            for idx, element in enumerate(org):
                cv2.line(img, (top_dev[idx], 0), (element, mid), 255, 2)
                cv2.line(img, (element, mid), (bottom_dev[idx], height), 255, 2)

            plt.imshow(X=img, cmap='gray')
            plt.title('image number: %d' % int(img_num))
            plt.xticks([])
            plt.yticks([])
            plt.show()

            # cv2.imwrite('./extracted-images/%d/L.bmp' % (int(img_num)), img)

            break
