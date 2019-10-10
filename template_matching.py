import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import time


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
# img = cv2.resize(src=img, dsize=(1682, 606))
rotated_img = copy.deepcopy(x=img)
print('image shape:', img.shape)
template = cv2.imread('./test-images/t7_cropped.jpg', 0)
print('template shape:', template.shape)
template_h, template_w = template.shape[:2]
img_h, img_w = img.shape[:2]
mid_h, mid_w = int(img_h/2.), int(img_w/2.)

"""best value template matching"""
# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# top_left = max_loc
# bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
# cv2.rectangle(img, top_left, bottom_right, 0, 2)
"""end of best value template matching"""


"""multiple values"""
# threshold = 0.55
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
"""end of multiple values"""


"""scale independent"""
# found = None
# for scale in np.linspace(1, 2, 4)[::-1]:
#     resized_template = cv2.resize(template, (int(template_w * scale), int(template_h * scale)))
#     # ratio = float(resized_template.shape[1]) / template_w
#     if resized_template.shape[0] > img_h or resized_template.shape[1] > img_w:
#         continue
#     res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     if found is None or max_val > found[0]:
#         found = (max_val, max_loc)
#     print('found is:', found)
#     (_, max_loc) = found
#     start_point = (int(max_loc[0] * scale), int(max_loc[1] * scale))
#     end_point = (int((max_loc[0] + template_w) * scale), int((max_loc[1] + template_h) * scale))
#     cv2.rectangle(img, start_point, end_point, 0, 2)
"""end of scale independent"""


"""rotation and scale invariant"""
t0 = time.time()
found = None
for degree in np.linspace(start=0, stop=360, num=46):

    # rotated_template, M = rotate_and_scale(image=template, scale=1.0, angle=degree)
    rotated_img, M = rotate_and_scale(image=rotated_img, scale=1.0, angle=degree)
    # if rotated_template.shape[0] > img_h or rotated_template.shape[1] > img_w:
    #     continue

    """single"""
    res = cv2.matchTemplate(rotated_img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    found = (max_val, max_loc)
    print('found is:', found)
    print('degree is:', degree)
    if max_val >= 0.7:
        org_img = copy.deepcopy(x=img)
        rotated_org_img, _ = rotate_and_scale(image=org_img, scale=1.0, angle=degree)
        sp = (int(max_loc[0]), int(max_loc[1]))
        ep = (int(max_loc[0] + template_w), int(max_loc[1] + template_h))
        cv2.rectangle(rotated_img, sp, ep, 0, 2)
        plt.imshow(X=rotated_org_img[sp[1]:ep[1], sp[0]:ep[0]], cmap='gray')
        plt.show()
    big_img, _ = rotate_and_scale(image=rotated_img, scale=1.0, angle=(360 - degree))
    big_img_h, big_img_w = big_img.shape[:2]
    rotated_img = big_img[int((big_img_h / 2.) - mid_h)+1:int((big_img_h / 2.) + mid_h),
                          int((big_img_w / 2.) - mid_w)+1:int((big_img_w / 2.) + mid_w)]
    # plt.imshow(X=rotated_img, cmap='gray')
    # plt.show()
#
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
t1 = time.time()
print('elapsed time: %.2f' % (t1-t0))
plt.imshow(rotated_img, cmap='gray')
plt.show()
# cv2.imwrite('./results/ritm_final.bmp', rotated_img)
