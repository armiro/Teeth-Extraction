import csv
import cv2
import numpy as np


img = cv2.imread('./test-images/7_lower.bmp', 0)
csv_file = open(file='./test-images/Tooth.csv', mode='r', newline='')
coordinates = csv.reader(csv_file, delimiter=',', quotechar='"')
teeth = list()
for row in coordinates:
    teeth.append(row)

top_coordinates = teeth[0]
middle_coordinates = teeth[1]
bottom_coordinates = teeth[2]

for idx in range(0, len(top_coordinates)):
    print(top_coordinates[idx], middle_coordinates[idx], bottom_coordinates[idx])
