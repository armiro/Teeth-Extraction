import cv2
import copy
import csv


csv_header = ['image_number', 'top_left_corner', 'bottom_right_corner']
csv_path = "./real_roi_corners.csv"
this_roi_corner_points = list()


def initialize_csv_reader(path):
    csv_file = open(path, mode='a', newline='')
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    writer.writerow(csv_header)
    return writer, csv_file


def append_to_csv(record, csv_file, writer):
    writer.writerow(record)
    csv_file.flush()


def draw_rectangle(event, x, y, flags, param):
    global this_roi_corner_points
    this_img = copy.deepcopy(x=img)

    if event == cv2.EVENT_LBUTTONDOWN:
        this_roi_corner_points = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        this_roi_corner_points.append((x, y))
        cv2.rectangle(this_img, this_roi_corner_points[0], this_roi_corner_points[1], 255, 3)
        cv2.imshow('image number %d' % i, this_img)

    # confirm the ROI with right-click
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyWindow('image number %d' % i)
        print("roi corner points of image %d are:" % i, this_roi_corner_points)
        this_record = [i, this_roi_corner_points[0], this_roi_corner_points[1]]
        append_to_csv(record=this_record, csv_file=csv_file, writer=writer)


writer, csv_file = initialize_csv_reader(path=csv_path)
for i in range(1, 51):
    img = cv2.imread('./images/%d.bmp' % i, 0)
    cv2.namedWindow('image number %d' % i, cv2.WINDOW_FREERATIO)
    cv2.imshow('image number %d' % i, img)
    cv2.setMouseCallback('image number %d' % i, draw_rectangle)
    cv2.waitKey(0)
