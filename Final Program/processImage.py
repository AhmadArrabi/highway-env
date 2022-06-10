import cv2 as cv
import numpy as np
from getDistance import getDistance

# print(f"///////////////////////////////////\ncenter = {center}")
   # print(f"center = {center}")
    # print(f"goal = {goals[2]}")
    # print(f"norm = {np.linalg.norm(center - goals[2])}\n+++++++++++++")
                # print(f"goal = {goal}")
                        # print(f"D = {D}\n+++++++++++++")
            #print(f"parkNum {parkNum}")



def isCar(goals,rect):
    valid = True
    center = np.array([rect[0][0]-470,rect[0][1]-240])
    i=0
    for goal in goals:
        # if(parkNum!=-1 and parkNum != None):
        #     D = getDistance(center[0],goal[0],center[1],goal[1])
        #     if (D < 60 and i != parkNum): #np.linalg.norm(center - goal)
        #         valid = False
        #         break
        # else:
            D = getDistance(center[0],goal[0],center[1],goal[1])
            if (D < 60):
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
    # print(f"len contours = {len(contours)}")
    if len(contours) >= 1:
        result = []
        resultParkings = []
        for car in contours:
            rect = cv.minAreaRect(car)
            box_ = cv.boxPoints(rect)
            box_ = np.int0(box_)
            valid,i = isCar(goals,rect)
            # print(f"newCNT: {valid}, {rect[1][0]*rect[1][1]}")
            if valid:
                temp = [box_,rect]
                result.append(temp)
            else:
                if num != None:
                    if i == num:
                        temp = [box_,rect]
                        result.append(temp)  
                resultParkings.append(i)
        return result ,resultParkings
    else:
        print("No Cars!")
        return None








# def checkParkings(contours,goals):
#     if len(contours) >= 1:
#         result = []
#         for car in contours:
#             rect = cv.minAreaRect(car)
#             box_ = cv.boxPoints(rect)
#             box_ = np.int0(box_)
#             valid,i = isCar(goals,rect,-1)
#             if not valid:
#                 result.append(i)
#         return result

# frame = cv.imread("./TEMP/1.png")
# refImage = cv.imread("./TEMP/ref_image.png")
# frame = cv.resize(frame,(640,480))
# refImage = cv.resize(refImage,(640,480))
# s = cv.subtract(frame,refImage)
# c = processing(s,15,7,True)
# parking1 = [(74,209),(386,478)]
# for ci in c:
#     finalCnts = q
#     for q in ci:
#         x_in = q[0][0]>=74 and q[0][0] <=209
#         y_in = q[0][1]>=386 and q[0][1] <=478
#         if (x_in or y_in):
#             finalCnts = np.delete(finalCnts,q,0)
#     print(finalCnts)
#     finalCnts  = np.array(finalCnts)
#     cv.drawContours(frame,[finalCnts],0,(0,255,0),5)
# cv.imshow("d",frame)
# cv.waitKey(0)