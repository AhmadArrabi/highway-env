import cv2 as cv
from chooseParking import chooseParking
from goTo import goTo
from mapAction import mapAction
from transform import four_point_transform
import numpy as np
from processImage import processing, processing_Conts
from settingPoints import setPoints
from getPixels import getPixels, returnPixels
from getObsv import getObrs
from goodFrame import goodFrame
import time
import os.path
import socket 
from model import getAction_Server
from findGoalPos import findCenters
from findGoalPos import findXY_offset
##################3
import os
from stable_baselines3 import PPO
model_name = "Task_1_100_freq" 
model_path = os.path.join("models", model_name)
model_name_2 = "Task_2_50_freq" 
model_path_2 = os.path.join("models", model_name_2)
model_name_3 = "Task_3_50_freq_2" 
model_path_3 = os.path.join("models", model_name_3)

Task_1 = PPO.load(f"{model_path}/195000") #ORIGNAL
#Task_1 = PPO.load(f"{model_path}/99000")

Task_2 = PPO.load(f"{model_path_2}/Second_training_240000") #ORIGNAL
#Task_2 = PPO.load(f"{model_path_2}/79000") 
#Task_2 = PPO.load(f"{model_path_2}/99000") 

Task_3 = PPO.load(f"{model_path_3}/145000")

##############
HEADER = 64
PORT = 5050
SERVER = "192.168.1.100"
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
server.bind(ADDR)
ptime = time.time()
ctime = time.time()


points = np.array([(-1,-1),(-1,-1),(-1,-1),(-1,-1)], dtype = "float32") #must change
refImage = None
#must change
INIT = False
valid_car = False
firstFrame = True
frontRight = None
frontLeft = None
backRight = None
backLeft = None
isParking = False
choosePark = True
startModelServer = False
STOP = False
letsCheckParkings = True
doIt_y_goto = False
#
DEBUG = True
currentCords = [0,0,0,0]
steps = 0
i=0
done_parkings=0
temp_parking = []
j=0
o=0
goals = None
numOfFrames = 0
fps = 0
offset = None
refVector = [150,0]
anotherCars = False
revGo_to_y = False

full = None
#####size = input("Enter the size for the win: ").split(" ")
#no_parking = int(input("How many parkings? "))
no_parking = 6
parkings_cords = []
size = ["640","480"]
sizeOffset = 1
size[0] = int(size[0])*sizeOffset
size[1] = int(size[1])*sizeOffset
def draw_parkings(img,num):
    i=0
    for parking in parkings_cords:
        if i == num:
            cv.rectangle(img,parking[3],parking[1],(0,255,0),3)
        i+=1
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
parkingNumber = None
def handle_client(conn, addr):
    global choosePark,parkingNumber, doIt_y_goto,STOP, steps, numOfFrames, INIT,ptime,ctime,points,refImage,firstFrame,frontRight,frontLeft,backRight,backLeft,goals,offset,full,isParking
    global revGo_to_y,valid_car, anotherCars, letsCheckParkings, DEBUG,currentCords,startModelServer,steps,i,done_parkings,temp_parking,j,o,numOfFrames,fps,refVector,no_parking,parkings_cords,size,sizeOffset
    print(f"[NEW CONNECTION] {addr} connected.")
    goals = findCenters(parkings_cords)
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$##
    parkings_full = {}
    for x in range(len(goals)):
        parkings_full[x] = True
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$##
    connected = True
    while(connected):
        #time.sleep(0.5)
        ret, frame = video.read()
        #1- Saving ref Image
        #transform and resizing
        #frame = cv.resize(frame,(size[0],size[1]))
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
                #print(parkings_cords)
                with open("./TEMP/parkings","w") as file:
                    for parking in parkings_cords:
                        for parkingPoint in parking:
                            file.write(f"{parkingPoint[0]},{parkingPoint[1]}\t")
                        file.write("\n")
        if(steps > 0):
            #print(steps)
            done = False
            refImage = cv.imread("./TEMP/ref_image.png")
            refImage = cv.resize(refImage,(size[0],size[1]))
            if(steps >= 2):
                if steps >= 3:
                    refImage = four_point_transform(refImage,points)
                    refImage = cv.resize(refImage,(size[0],size[1]))
                    if letsCheckParkings:
                        frame = four_point_transform(frame,points)
                        frame = cv.resize(frame,(size[0],size[1]))
                    else:
                        #somthing else
                        frame = four_point_transform(frame,points)
                        frame = cv.resize(frame,(size[0],size[1]))
                        cv.rectangle(frame,(0,parkings_cords[0][1][1]),(parkings_cords[0][2][0],480),(0,0,0),-1) #bot
                        cv.rectangle(frame,(0,0),parkings_cords[1][2],(0,0,0),-1) #top
                        cv.rectangle(frame,(parkings_cords[5][0][0],0),(parkings_cords[5][3][0]+20,parkings_cords[5][3][1]),(0,0,0),-1)
                        cv.rectangle(frame,(parkings_cords[4][0][0],parkings_cords[4][0][1]-5),(parkings_cords[4][3][0]+20,parkings_cords[4][3][1]),(0,0,0),-1)
                        blackBoxes = [1]*6
                        blackBoxes[parkingNumber] = 0
                        if blackBoxes[0]:
                            cv.rectangle(frame,parkings_cords[0][1],(parkings_cords[2][2][0],480),(0,0,0),-1) #parking 0
                        if blackBoxes[2]:
                            cv.rectangle(frame,parkings_cords[0][0],(parkings_cords[4][2][0],480),(0,0,0),-1) #parking 2
                        if blackBoxes[4]:
                            cv.rectangle(frame,parkings_cords[2][0],(parkings_cords[4][3][0]+20,480),(0,0,0),-1) #parking 4
                        if blackBoxes[1]:
                            cv.rectangle(frame,(parkings_cords[1][1][0],0),parkings_cords[3][2],(0,0,0),-1) #parking 1
                        if blackBoxes[3]:
                            cv.rectangle(frame,(parkings_cords[1][0][0],0),parkings_cords[5][2],(0,0,0),-1) #parking 3
                        if blackBoxes[5]:
                            cv.rectangle(frame,(parkings_cords[3][0][0],0),parkings_cords[5][3],(0,0,0),-1) #parking 5
                    subtacted_frame = cv.subtract(frame,refImage)
                    if parkingNumber != None:
                        draw_parkings(frame,parkingNumber)
                    if DEBUG:
                        cv.imshow("sub",subtacted_frame)
                    conts = processing(subtacted_frame,15,5,DEBUG)
                    results,fullParkings = processing_Conts(conts,goals,parkingNumber)
                    if len(fullParkings) > 5:
                        print("No parkings Av.")
                        STOP = True
                        break
                    if letsCheckParkings:
                        letsCheckParkings = False
                        if len(fullParkings)>0:
                            for p in parkings_full.keys():
                                if p in fullParkings:
                                    parkings_full[p] = False
                                else:
                                    parkings_full[p] = True
                    result = None
                    otherObjects = []
                    if (results != None):
                        if len(results) < 1:
                            print("No New Cars!")
                        elif len(results) == 1:
                            result = results[0]
                        elif len(results) > 1:
                            result,otherObjects = goodFrame(results)
                    if choosePark:
                        parkingNumber = chooseParking(parkings_full,result,goals)
                        choosePark = False
                        continue
                    if len(result) < 1:
                        print("No valid Car, make sure that the car in the parking")
                        valid_car = False
                        STOP = True
                        #conn.send(f"{str(10)},{'No cars'}".encode(FORMAT))
                    else:
                        valid_car = True
                    print(f"len others = {len(otherObjects)}")
                    if len(otherObjects) >= 1:
                        THRESH = 1000
                        for ob in otherObjects:
                            box = ob[1][1]
                            area = box[0] * box[1]
                            print(f"area = {area}")
                            if area > THRESH:
                                print("The env is not safe, please make sure that the street is empty!")
                                STOP = True
                                break
                    if(not parkings_full[parkingNumber] and not isParking):
                        print("Parking is full!")
                        STOP = True
                    else:
                        isParking = True
                    if(True):
                        numOfFrames += 1
                        if(result):
                            if not INIT:
                                print("IN")
                                TEMP_POINTS = result[0]
                                allX = TEMP_POINTS[:,0]
                                allY = TEMP_POINTS[:,1]
                                backLeft = (min(allX),min(allY))
                                backRight = (min(allX),max(allY))
                                frontLeft = (max(allX),min(allY))
                                frontRight = (max(allX),max(allY))
                                ####### GOTO Y #########
                                currentY = (result[1][0][1] -240)
                                #print(currentY)
                                if(currentY <= 0):
                                    if parkingNumber%2 == 0:
                                        doIt_y_goto = True
                                    else:
                                        doIt_y_goto = False
                                else:
                                    if parkingNumber%2 == 0:
                                        doIt_y_goto = False
                                    else:
                                        doIt_y_goto = True
                                ########################
                                #print(f"{frontRight}\t{frontLeft}\t{backRight}\t{backLeft}\n$$$$$$$$")
                                INIT = True
                            oldPoints = [frontRight,frontLeft,backRight,backLeft]
                            newPoints = result[0]
                            pos = result[1][0]
                            if firstFrame:
                                newPoints = np.array(oldPoints)
                                firstFrame = False
                            [frontRight,frontLeft,backRight,backLeft] = setPoints(oldPoints,newPoints)
                            rev = True
                            if rev:
                                currentCords[0] = backLeft
                                currentCords[1] = backRight
                                currentCords[2] = frontRight
                                currentCords[3] = frontLeft
                            else:
                                currentCords[2] = backLeft
                                currentCords[3] = backRight
                                currentCords[0] = frontRight
                                currentCords[1] = frontLeft
                            newVector = [frontRight[0]-backRight[0],frontRight[1]-backRight[1]]
                            offset = findXY_offset(goals[parkingNumber],parkingNumber)
                            obsrevs,overAllOffset = getObrs(currentCords,parkings_cords[parkingNumber],parkingNumber,refVector,newVector,frontRight,backRight,pos[0],pos[1],offset)
                            if not valid_car:
                                conn.send(f"{str(10)},{'No cars'}".encode(FORMAT))
                            #print(obsrevs)
                            #Model######
                            startModel,forWard,curve,y_status = goTo(obsrevs,overAllOffset,parkings_cords[parkingNumber],parkingNumber,goals,doIt_y_goto)
                            if startModel:
                            #if True:
                                startModelServer = True
                            if startModelServer:
                                # action = 10
                                # done = True
                                # task = "TEMP"
                               action,done,task = getAction_Server(obsrevs,Task_1,Task_2,Task_3,goals[parkingNumber],overAllOffset)
                            elif not startModelServer:
                                if not y_status[2]:
                                    if result[1][0][0] >= goals[parkingNumber][0] and result[1][0][0] >= 320:
                                        revGo_to_y = True
                                    if y_status[0] == False:
                                        if parkingNumber %2 != 0:
                                            # if revGo_to_y:
                                            #     action = mapAction(3)
                                            # else:
                                            action = mapAction(2)
                                        else:
                                            # if revGo_to_y:
                                            #     action = 15
                                            # else:
                                            action = 20
                                    elif y_status[0] and not y_status[1]:
                                        if parkingNumber %2 != 0:
                                            # if revGo_to_y:
                                            #     action = mapAction(18)
                                            # else:
                                            action = mapAction(17)
                                        else:
                                            # if revGo_to_y:
                                            #     action = 0
                                            # else:
                                            action = 5
                                elif y_status[2]:
                                    if not curve:
                                        if forWard:
                                            action = 11
                                        else:
                                            action = 9
                                    else:
                                        action = 20
                                task = "Go To INIT"
                            if parkingNumber%2 == 0:
                                #print(action)
                                conn.send(f"{str(action)},{str(task)}".encode(FORMAT))
                                #conn.send(f"{str(11)},{str(task)}".encode(FORMAT))
                            else:
                                action_ = mapAction(action)
                                #print(action_)
                                conn.send(f"{str(action_)},{str(task)}".encode(FORMAT))
                                #conn.send(f"{str(11)},{str(task)}".encode(FORMAT))
                            #conn.send(f"{str(11)},{str(task)}".encode(FORMAT))
                            ############
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
        #cv.putText(frame,str(int(fps)),(10,40),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
        cv.imshow("image with black boxes",frame)
        cv.setMouseCallback('image with black boxes',getPixels,frame)
        if (cv.waitKey(25) & 0xFF == ord('q')) or done or STOP:
            connected = False
            video.release()
            conn.send("!D".encode(FORMAT))
            cv.waitKey(0)
            cv.destroyAllWindows()
def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    conn, addr = server.accept()
    # parking = input("Enter parking number:")
    # #parking = 0
    # try:
    #     parking = int(parking)
    # except:
    #     print("NOT VALID!")
    #     conn.send("!D".encode(FORMAT))
    #     return
    try:
        handle_client(conn,addr)
    except Exception as e:
        print("Something Went Wrong!")
        print(e)
        conn.send("!D".encode(FORMAT))
    finally:
        conn.close()


start()