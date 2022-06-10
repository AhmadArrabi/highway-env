done1 = False
done2 = False
import numpy as np
def goToY(obs,overAllOffset,goals,parkingNumber,doIt):
    global done1,done2
    sign = 1
    if parkingNumber%2 != 0:
        sign = -1
    parkingY = goals[parkingNumber][1]
    currentY = (sign*(obs['Position'][1])-overAllOffset[1])
    if(doIt):
        desY = parkingY + (180 *(-sign)) #180 full battery , 160 76%
        diff = desY - currentY
        THESH = 20
        heading = obs['Heading'][0]
        #print(f"done1 = {done1}   done2 = {done2}   diff={diff}   heading = {heading}")
        if not done1:
            if np.abs(diff) <= THESH:
                done1 = True
        elif done1 and not done2:
            if np.abs(heading) <= 10: #60% -> 5
                done2 = True
    else:
        #print("Don't Go Y")
        done1 = True
        done2 = True
    return done1,done2
