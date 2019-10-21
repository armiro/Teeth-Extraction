import cv2
import matplotlib.pyplot as plt
import glob

upper_text_path = './upper_jaws/up.txt'
upper_text_file = open(file=upper_text_path, mode='r')
lines = upper_text_file.readlines()

upper_rev_path = './upper_jaws/up_revision.txt'
upper_rev_file = open(file=upper_rev_path, mode='r')
rev_indices = upper_rev_file.readlines()

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

for image_name in glob.glob(pathname="./upper_jaws/**.bmp"):
    img_num = image_name.split('\\')[-1].split('_')[0]
    print("image number is:", img_num)
    img = cv2.imread(image_name, 0)
    height, width = img.shape[:2]
    upsize_coef = round((width / 216), ndigits=2)
    print('up-size coefficient=', upsize_coef)
    rev = [e[1] for e in rev_pts if e[0] == img_num][0]
    rev = [int(e) for e in rev]

    for pt in pts:
        if pt[0] == img_num:
            print("point found:", pt[0])
            coordinates = [int(int(point) * upsize_coef) for point in pt[1]]
            print('initial:', coordinates)

            coordinates = [coordinates[i - 1] for i in rev]
            coordinates = [0] + coordinates + [width]
            print('final:', coordinates)
            print('num lines:', len(coordinates))
            # draw the lines on the initial image and save it
            for idx, element in enumerate(coordinates):
                cv2.line(img, (coordinates[idx], 0), (element, height), 255, 2)

            plt.imshow(X=img, cmap='gray')
            plt.title(label='%d' % int(img_num))
            plt.show()

            cv2.imwrite('./extracted-images/%d/H.bmp' % (int(img_num)), img)
            break
