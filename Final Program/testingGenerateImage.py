import cv2 as cv
import numpy as np
from transform import four_point_transform
o = 0
points  = np.array([(-1,-1),(-1,-1),(-1,-1),(-1,-1)], dtype = "float32")
with open("./TEMP/4pointsTrans","r") as file:
            _points = file.read().split("\t")
            _points = _points[:-1]
            for point in _points:
                point = point.split(",")
                points[o] = (float(point[0]),float(point[1]))
                o+=1
parkings_cords = []
with open("./TEMP/parkings","r") as file:
                parkings = file.read().split("\n")
                parkings = parkings[:-1]
                for parking in parkings:
                    temp = []
                    parking = parking.split("\t")
                    parking = parking[:-1]
                    for parkingPoints in parking:
                        parkingPoints = parkingPoints.split(",")
                        temp.append((int(parkingPoints[0]),int(parkingPoints[1])))
                    parkings_cords.append(temp)
ref = cv.imread("./TEMP/ref_image.png")
ref = four_point_transform(ref,points)
frame = cv.imread("./TEMP/frame.png")
frame = four_point_transform(frame,points)
frame = cv.resize(frame,(640,480))
ref = cv.resize(ref,(640,480))
cv.imshow("s1", frame)
goal = 2
blackBoxes = [1]*6
blackBoxes[goal] = 0
if blackBoxes[0]:
    cv.rectangle(frame,(0,parkings_cords[0][1][1]),parkings_cords[0][3],(0,0,0),-1) #parking 0
if blackBoxes[2]:
    cv.rectangle(frame,parkings_cords[0][0],parkings_cords[2][3],(0,0,0),-1) #parking 2
if blackBoxes[4]:
    cv.rectangle(frame,parkings_cords[2][0],parkings_cords[4][3],(0,0,0),-1) #parking 4
if blackBoxes[1]:
    cv.rectangle(frame,(0,parkings_cords[1][1][1]),parkings_cords[1][3],(0,0,0),-1) #parking 1
if blackBoxes[3]:
    cv.rectangle(frame,parkings_cords[1][0],parkings_cords[3][3],(0,0,0),-1) #parking 3
if blackBoxes[5]:
    cv.rectangle(frame,parkings_cords[3][0],parkings_cords[5][3],(0,0,0),-1) #parking 5

cv.imshow("s2", frame)
subtacted_frame = cv.subtract(frame,ref)
cv.imshow("s3", subtacted_frame)
cv.waitKey(0)