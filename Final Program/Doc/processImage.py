import cv2 as cv
import numpy as np
from pyzbar import pyzbar


def isCar(goals,rect,parkNum):
    valid = True
    center = np.array([rect[0][0]-470,rect[0][1]-240])
    i=0
    for goal in goals:
        if(parkNum!=-1):
            if (np.linalg.norm(center - goal) < 25 and i != parkNum):
                valid = False
                break
        else:
            if (np.linalg.norm(center - goal) < 25):
                valid = False
                break
        i+=1
    return valid,i


def processing(sub,dIT,eIT,DEBUG):
    gray = cv.cvtColor(sub,cv.COLOR_BGR2GRAY)
    if DEBUG:
        cv.imshow("gray sub",gray)
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((6, 6), np.uint8)
    (_,thers) = cv.threshold(gray,75,255,cv.THRESH_BINARY)
    blur = cv.dilate(thers, kernel, iterations=dIT)
    if DEBUG:
        cv.imshow("Dilate",blur)
    blur = cv.erode(blur, kernel2, cv.BORDER_WRAP, iterations=eIT)
    if DEBUG:
        cv.imshow("Erode",blur)
    contours, hirearchies = cv.findContours(blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return contours
def processing_Conts(contours,goals,num):
    if len(contours) >= 1:
        result = []
        for car in contours:
            rect = cv.minAreaRect(car)
            box_ = cv.boxPoints(rect)
            box_ = np.int0(box_)
            valid,i = isCar(goals,rect,num)
            if valid:
                temp = [box_,rect]
                result.append(temp)
        return result
    else:
        print("No Cars!")
        return None
def checkParkings(contours,goals):
    if len(contours) >= 1:
        result = []
        for car in contours:
            rect = cv.minAreaRect(car)
            box_ = cv.boxPoints(rect)
            box_ = np.int0(box_)
            valid,i = isCar(goals,rect,-1)
            if not valid:
                result.append(i)
        return result