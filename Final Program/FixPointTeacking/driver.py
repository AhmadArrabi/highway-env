import cv2 as cv
import numpy as np
from FixPointTeacking.getXY import getXY
from FixPointTeacking.tracker import CentroidTracker

def processing2(sub,dIT,eIT):
    ct = CentroidTracker()
    rects = []
    #####################################
    gray = cv.cvtColor(sub,cv.COLOR_BGR2GRAY)
    kernel = np.ones((6, 6), np.uint8)
    (_,thers) = cv.threshold(gray,75,255,cv.THRESH_BINARY)
    blur = cv.dilate(thers, kernel, iterations=dIT)
    cv.imshow("image3",thers)
    blur = cv.erode(blur, kernel, cv.BORDER_WRAP, iterations=eIT)
    cv.imshow("image2",blur)
    contours, hirearchies = cv.findContours(blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) <1:
        print("No car yet!")
    else:
        rect = cv.minAreaRect(contours[0])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        rects.append(getXY(box))
    #####################################
    objects = ct.update(rects)
    return (objects,box)