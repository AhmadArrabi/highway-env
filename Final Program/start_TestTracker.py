import cv2 as cv
from transform import four_point_transform
import numpy as np
from processImage import processing
from settingPoints import setPoint
from getPixels import getPixels, returnPixels
from getObsv import getObrs
from goodFrame import goodFrame
from FixPointTeacking import driver
import time
import os.path

ptime = time.time()
ctime = time.time()

points = np.array([(-1,-1),(-1,-1),(-1,-1),(-1,-1)], dtype = "float32") #must change
refImage = None
#must change
INIT = False
frontRight = None
frontLeft = None
backRight = None
backLeft = None
#
currentCords = [0,0,0,0]
steps = 0
i=0
done_parkings=0
temp_parking = []
j=0
o=0
numOfFrames = 0
fps = 0 
#####size = input("Enter the size for the win: ").split(" ")
#no_parking = int(input("How many parkings? "))
no_parking = 8
parkings_cords = []
size = ["640","480"]
size[0] = int(size[0])
size[1] = int(size[1])
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
            if done_parkings < no_parking:
                point = returnPixels()
                temp_parking.append(point)
                print(f"{point}\tPoint saved!")
                j+=1
                if j > 3:
                    parkings_cords.append(temp_parking)
                    temp_parking = []
                    done_parkings += 1
                    if done_parkings >= no_parking:
                        steps += 1
                        with open("./TEMP/parkings","w") as file:
                            for parking in parkings_cords:
                                for parkingPoint in parking:
                                    file.write(f"{parkingPoint[0]},{parkingPoint[1]}\t")
                                file.write("\n")
                    j=0
    if(steps > 0):
        #print(steps)
        refImage = cv.imread("./TEMP/ref_image.png")
        refImage = cv.resize(refImage,(size[0],size[1]))
        if(steps >= 2):
            refImage = four_point_transform(refImage,points)
            frame = four_point_transform(frame,points)
            if steps >= 3: 
                subtacted_frame = cv.subtract(frame,refImage)
                (objects,box) = driver.processing2(subtacted_frame,8,3)
                for (objectID, centroid) in objects.items():
                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = "ID {}".format(objectID)
                    cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                cv.drawContours(frame,[box],0,(0,255,255),2)
            else:
                print(f"{no_parking - done_parkings} parkings left!\tStep: {steps}")    
        else:
            print(f"{4-i} points left\tStep: {steps}")
    else:
        print("Get ref image")
        #distances = getObrs(currentCords,parking1)
    #6- handling get pixels
    ctime = time.time()
    if ctime - ptime >= 1:
        fps = numOfFrames
        ptime = ctime
        numOfFrames = 0
    cv.putText(frame,str(int(fps)),(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
    cv.imshow("image",frame)
    cv.setMouseCallback('image',getPixels,frame)
    #print(returnPixels())
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
video.release()
cv.waitKey(0)
cv.destroyAllWindows()