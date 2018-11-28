# import numpy as np
import cv2
import sys
import argparse


def splitImage(inImg):
    h, w = inImg.shape[0], inImg.shape[1]
    off1X = 0
    off1Y = 0
    off2X = 0
    off2Y = 0

    if w >= h:  # split X
        off1X = 0
        off2X = w / 2
        img1 = inImg[0:int(h), 0:int(off2X)]
        img2 = inImg[0:int(h), int(off2X):int(w)]
    else:  # split Y
        off1Y = 0
        off2Y = h / 2
        img1 = inImg[0:int(off2Y), 0:int(w)]
        img2 = inImg[int(off2Y):int(h), 0:int(w)]

    return off1X, off1Y, img1, off2X, off2Y, img2


class theQTree:

    def qt(self, inImg, minStd, minSize, offX, offY):
        h, w = inImg.shape[0], inImg.shape[1]
        m, s = cv2.meanStdDev(inImg)

        if s >= minStd and max(h, w) > minSize:
            oX1, oY1, im1, oX2, oY2, im2 = splitImage(inImg)

            self.qt(im1, minStd, minSize, offX + oX1, offY + oY1)
            self.qt(im2, minStd, minSize, offX + oX2, offY + oY2)
        else:
            self.listRoi.append([offX, offY, w, h, m, s])

    def __init__(self, img, stdmin, sizemin):
        self.listRoi = []
        self.qt(img, stdmin, sizemin, 0, 0)


offx = 0
offy = 0


def quadtree(inImg, minStd, minSize, offX, offY, roiList):
    h, w = inImg.shape[0], inImg.shape[1]
    m, s = cv2.meanStdDev(inImg)

    if s >= minStd and max(h, w) > minSize:
        oX1, oY1, im1, oX2, oY2, im2 = splitImage(inImg)

        quadtree(im1, minStd, minSize, offX + oX1, offY + oY1, roiList)
        quadtree(im2, minStd, minSize, offX + oX2, offY + oY2, roiList)
    else:
        roiList.append([offX, offY, w, h, m, s])


def main(data_):
    parser = argparse.ArgumentParser(description='Compute best bars cuts')

    parser.add_argument('-img',
                        metavar='',
                        type=str,
                        help='image to load')

    parser.add_argument('-sz',
                        metavar='',
                        type=int,
                        help='quadtree min size')

    parser.add_argument('-std',
                        metavar='',
                        type=float,
                        help='standard deviation to split')

    args = parser.parse_args()

    # input values
    IMAGE_TO_LOAD = args.img

    minDev = args.sz
    minSz = args.std

    raw = cv2.imread(IMAGE_TO_LOAD)

    if not type(raw) == type(None):
        if raw.ndim > 1:
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            img = raw
    else:
        print('Error on input image: ', IMAGE_TO_LOAD)
        exit()

    print('execution...')

    cv2.imshow('Start Image', img)
    h, w = img.shape[0], img.shape[1]

    # if color img, sequence bgr
    m, s = cv2.meanStdDev(img)

    qt = theQTree(img, minDev, minSz)
    rois = qt.listRoi

    imgOut = img
    for e in rois:
        for element in range(0, len(e)):
            e[element] = int(e[element])
        col = 255
        if e[5] < minDev:
            col = 0

        cv2.rectangle(imgOut, (e[0], e[1]), (e[0] + e[2], e[1] + e[3]), col, 1)

    cv2.imshow('Quad Image', imgOut)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1])
