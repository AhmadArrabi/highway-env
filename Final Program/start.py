import cv2 as cv
from goTo import goTo
from transform import four_point_transform
import numpy as np
from processImage import processing, processing_Conts
from HeadingClac import calcAngle
from settingPoints import setPoints
from getPixels import getPixels, returnPixels
from getObsv import getObrs
from goodFrame import goodFrame
import time
import os.path
from findGoalPos import *

goals = None
ptime = time.time()
ctime = time.time()

points = np.array([(-1,-1),(-1,-1),(-1,-1),(-1,-1)], dtype = "float32") #must change
refImage = None
#must change
INIT = False
firstFrame = True
frontRight = None
frontLeft = None
backRight = None
backLeft = None
#
full = None
DEBUG = True
currentCords = [0,0,0,0]
steps = 0
i=0
done_parkings=0
temp_parking = []
j=0
o=0
numOfFrames = 0
fps = 0
refVector = [150,0]
#####size = input("Enter the size for the win: ").split(" ")
#no_parking = int(input("How many parkings? "))
no_parking = 6
parkings_cords = []
size = ["640","480"]
sizeOffset = 1
size[0] = int(size[0])*sizeOffset
size[1] = int(size[1])*sizeOffset
def draw_parkings(img):
    for parking in parkings_cords:
        cv.rectangle(img,parking[3],parking[1],(0,255,0),3)
if (os.path.exists("./TEMP/ref_image.png")):
    if(os.path.exists("./TEMP/4pointsTrans")):
        with open("./TEMP/4pointsTrans","r") as file:
            _points = file.read().split("\t")
            _points = _points[:-1]
            for point in _points:
                point = point.split(",")
                points[o] = (float(point[0]),float(point[1]))
                o+=1
        if(os.path.exists("./TEMP/parkings")):
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
            steps = 3
        else:
            steps = 2
    else:
        steps = 1
else:
    steps = 0
video = cv.VideoCapture(2)

goals = findCenters(parkings_cords)
# goalPos = goals[0]
# #print(f"Real Image : {goalPos[0]+470}   {goalPos[1] +240}")
# print(f"Sim : {goalPos[0]}   {goalPos[1]}")
# offset = findXY_offset(goalPos)
# print(offset)
# print(f"Sim After Offset: {goalPos[0]+offset[0]}   {goalPos[1]+offset[1]}")
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$##
parkings_full = {}
for x in range(len(goals)):
        parkings_full[x] = True
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$##
while(True):
    ret, frame = video.read()
    #1- Saving ref Image
    #transform and resizing
    frame = cv.resize(frame,(size[0],size[1]))
    if cv.waitKey(25) & 0xFF == ord('s'):
        if steps ==0:
            #save ref image
            cv.imwrite("./TEMP/ref_image.png",frame)
            steps+=1
        if steps == 1:
            pix = returnPixels()
            points[i] = pix
            print(f"{points[i]}\tPoint saved!")
            i+=1
            if i > 3:
                steps += 1
                with open("./TEMP/4pointsTrans","w") as file:
                    for point in points:
                        file.write(f"{point[0]},{point[1]}\t")
        if steps == 2:
            img = four_point_transform(frame,points)
            img = cv.resize(img,(size[0],size[1]))
            cv.destroyWindow("image")
            for n in range(no_parking):
                r = cv.selectROI(img)
                parking_points = [(r[2]+r[0],r[1]),(r[0],r[1]),(r[0],r[3]+r[1]),(r[2]+r[0],r[3]+r[1])]
                parkings_cords.append(parking_points)
            steps += 1
            cv.destroyWindow("ROI selector")
            (parkings_cords)
            with open("./TEMP/parkings","w") as file:
                for parking in parkings_cords:
                    for parkingPoint in parking:
                        file.write(f"{parkingPoint[0]},{parkingPoint[1]}\t")
                    file.write("\n")
    if(steps > 0):
        #print(steps)
        refImage = cv.imread("./TEMP/ref_image.png")
        refImage = cv.resize(refImage,(size[0],size[1]))
        if(steps >= 2):
            refImage = four_point_transform(refImage,points)
            refImage = cv.resize(refImage,(size[0],size[1]))
            frame = four_point_transform(frame,points)
            frame = cv.resize(frame,(size[0],size[1]))
            if steps >= 3:
                subtacted_frame = cv.subtract(frame,refImage)
                draw_parkings(frame)
                if DEBUG:
                    cv.imshow("sub",subtacted_frame)
                conts = processing(subtacted_frame,15,5,DEBUG)
                results,fullParkings = processing_Conts(conts,goals,-1)
                #fullParkings = checkParkings(conts,goals)
                result = None
                if fullParkings != None:
                    for p in parkings_full.keys():
                        if p in fullParkings:
                            parkings_full[p] = False
                        else:
                            parkings_full[p] = True
                #print(parkings_full)
                if (results != None):
                    if len(results) < 1:
                        print("No New Cars!")
                    elif len(results) == 1:
                        result = results[0]
                    elif len(results) > 1:
                        result,ooooooo = goodFrame(results)
                #print(f"{result[0][0]}  {result[0][1]}  {result[0][2]}  {result[0][3]}.. {result[1][2]}")
                if(True):
                    numOfFrames += 1
                    if(result):
                        #print(result[1])
                        if not INIT:
                            print("IN")
                            #print(result[0])
                            TEMP_POINTS = result[0]
                            allX = TEMP_POINTS[:,0]
                            allY = TEMP_POINTS[:,1]
                            backLeft = (min(allX),min(allY))
                            backRight = (min(allX),max(allY))
                            frontLeft = (max(allX),min(allY))
                            frontRight = (max(allX),max(allY))
                            #print(f"{frontRight}\t{frontLeft}\t{backRight}\t{backLeft}\n$$$$$$$$")
                            INIT = True
                        oldPoints = [frontRight,frontLeft,backRight,backLeft]
                        newPoints = result[0]
                        if firstFrame:
                            newPoints = np.array(oldPoints)
                            firstFrame = False
                        [frontRight,frontLeft,backRight,backLeft] = setPoints(oldPoints,newPoints)
                        currentCords[0] = backLeft
                        currentCords[1] = backRight
                        currentCords[2] = frontRight
                        currentCords[3] = frontLeft
                        newVector = [frontRight[0]-backRight[0],frontRight[1]-backRight[1]]
                        #print(parkings_cords)
                        pos = result[1][0]
                        # print(goals[0])
                        # print(result[1][0][0]-470,result[1][0][1]-240)
                        
                        #   )
                        blank = frame
                        blank = cv.circle(blank, frontRight, radius=0, color=(255, 0, 0), thickness=8)
                        blank = cv.circle(blank, frontLeft, radius=0, color=(0, 255, 0), thickness=8)
                        blank = cv.circle(blank, backRight, radius=0, color=(0, 0, 255), thickness=8)
                        blank = cv.circle(blank, backLeft, radius=0, color=(255, 0, 255), thickness=8)
                        cv.drawContours(blank,[result[0]],0,(0,255,255),2)
            else:
                print(f"{no_parking - done_parkings} parkings left!\tStep: {steps}")    
        else:
            print(f"{4-i} points left\tStep: {steps}")
    else:
        print("Get ref image")
    #6- handling get pixels
    ctime = time.time()
    if ctime - ptime >= 1:
        fps = numOfFrames
        ptime = ctime
        numOfFrames = 0
    cv.putText(frame,str(int(fps)),(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
    cv.imshow("image",frame)
    cv.setMouseCallback('image',getPixels,frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
video.release()
cv.waitKey(0)
cv.destroyAllWindows()