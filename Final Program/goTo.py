import numpy as np

from goToY import goToY
counter = 0
FinalDone = False
doneMoving = False
doneY = False
forWard = True
y_status = [False,False,False]
def goTo(obs,overAllOffset,goalCords,parkingNumber, goals,doIt_y):
    global counter, FinalDone, doneMoving, doneY,forWard, y_status
    done1 , done2 = goToY(obs,overAllOffset,goals,parkingNumber,doIt_y)
    if done1 and done2:
        doneY = True
    y_status = [done1,done2,doneY]
    if doneY:
        limitCounter = (parkingNumber*2.5) + 5
        #print(f"counter = {counter}")
        forWard = True
        THESH = 20 #40 Top
        currentX = (obs['Position'][0])-overAllOffset[0]+470
        parkingX = min(goalCords)[0]
        desX = parkingX - 50 #50
        diff = desX - currentX
        if diff < 0 and not doneMoving:
            forWard = False
        if np.abs(diff) < THESH:
            doneMoving = True
        if forWard:
            if doneMoving:
                if parkingNumber %2 ==0:
                    if counter >=limitCounter:
                        FinalDone = True
                    counter +=1
                else:
                    FinalDone = True
        else:
            if doneMoving:
                FinalDone = True
    return FinalDone,forWard,doneMoving,y_status