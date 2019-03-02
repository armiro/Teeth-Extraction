import cv2
import numpy as np
from matplotlib import pyplot as plt


def rotate_and_scale(image, scale=1.0, angle=0):
    height, width = image.shape[:2]
    mat = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=angle, scale=scale)

    new_w, new_h = width * scale, height * scale
    r = np.deg2rad(angle)
    new_w, new_h = (abs(np.sin(r) * new_h) + abs(np.cos(r) * new_w), abs(np.sin(r) * new_w) + abs(np.cos(r) * new_h))
    (tx, ty) = ((new_w - width)/2., (new_h - height)/2.)
    mat[0, 2] += tx
    mat[1, 2] += ty

    rotated_image = cv2.warpAffine(src=image, M=mat, dsize=(int(new_w), int(new_h)))
    return rotated_image, mat


img = cv2.imread('./test-auto-cropped/2.bmp', 0)
print('image shape:', img.shape)
template = cv2.imread('./test-images/t2.jpg', 0)
print('template shape:', template.shape)
template_h, template_w = template.shape[:2]
img_h, img_w = img.shape[:2]

"""best value template matching"""
# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# top_left = max_loc
# bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
# cv2.rectangle(img, top_left, bottom_right, 0, 3)
"""end of best value template matching"""

"""scale independent"""
# found = None
# for scale in np.linspace(0.1, 1, 10)[::-1]:
#
#     resized_template = cv2.resize(template, (int(template_h * scale), int(template_w * scale)))
#     ratio = template_w / float(resized_template.shape[1])
#     if resized_template.shape[0] > img_h or resized_template.shape[1] > img_w:
#         continue
#     print(resized_template.shape)
#     res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     if found is None or max_val > found[0]:
#         found = (max_val, max_loc, ratio)
#     print('found is:', found)
#     (_, max_loc, ratio) = found
#     start_point = (int(max_loc[0] * ratio), int(max_loc[1] * ratio))
#     end_point = (int((max_loc[0] + template_w) * ratio), int((max_loc[1] + template_h) * ratio))
#     cv2.rectangle(img, start_point, end_point, (0, 0, 255), 2)
"""end of scale independent"""

"""multiple values"""
# threshold = 0.55
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
"""end of multiple values"""

"""rotation and scale invariant"""
found = None
for degree in np.linspace(start=20, stop=21, num=2):

    # rotated_template, M = rotate_and_scale(image=template, scale=1.0, angle=degree)
    rotated_img, M = rotate_and_scale(image=img, scale=1.0, angle=degree)
    print(rotated_img)
    print(type(rotated_img))
    print(len(rotated_img))
    # if rotated_template.shape[0] > img_h or rotated_template.shape[1] > img_w:
    #     continue

    print(degree)
    """single"""
    res = cv2.matchTemplate(rotated_img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if found is None or max_val > found[0]:
        found = (max_val, max_loc)
        print('found is:', found)
        print('degree is:', degree)
    (_, max_loc) = found
    sp = (int(max_loc[0]), int(max_loc[1]))
    ep = (int(max_loc[0] + template_w), int(max_loc[1] + template_h))
    cv2.rectangle(img, sp, ep, 0, 2)
    plt.imshow(X=rotated_img, cmap='gray')
    plt.show()

#     """multiple"""
#     res = cv2.matchTemplate(img, rotated_template, cv2.TM_CCOEFF_NORMED)
#     threshold = 0.7
#     loc = np.where(res >= threshold)
#     for pt in zip(*loc[::-1]):
#         cv2.rectangle(img, pt, (pt[0] + template_w, pt[1] + template_h), (0, 0, 255), 2)
#
#     top_left = sp
#     top_right = (sp[0] + template_w, sp[1] + template_h)
#     bottom_right = ep
#     center = (np.mean(a=[top_left[0], bottom_right[0]]), np.mean(a=[top_left[1], bottom_right[1]]))
#     top_left = (top_left[0] - center[0], top_left[1] - center[1])
#     top_right = (top_right[0] - center[0], top_right[1] - center[1])
#     bottom_right = (bottom_right[0] - center[0], bottom_right[1] - center[1])
#     rotation_matrix = M[:, 0:2]
#     top_left = np.array([[top_left[0]], [top_left[1]]])
#     top_right = np.array([[top_right[0]], [top_right[1]]])
#     bottom_right = np.array([[bottom_right[0]], [bottom_right[1]]])
#
#     top_left = np.matmul(rotation_matrix, top_left)
#     top_right = np.matmul(rotation_matrix, top_right)
#     bottom_right = np.matmul(rotation_matrix, bottom_right)




"""end of rotation and scale invariant"""


# mat = cv2.getRotationMatrix2D(center=(template_w / 2, template_h / 2), angle=10, scale=1.0)
# print(mat)
plt.imshow(img, cmap='gray')
plt.show()
