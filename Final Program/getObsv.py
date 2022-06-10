from getDistance import getDistance
from HeadingClac import calcAngle
import numpy as np

def getDistances(current,parking,bottom):
        distances = [0,0,0,0]
        if bottom:
            distances[0] = getDistance(current[0][0],parking[0][0],current[0][1],parking[0][1])
            distances[1] = getDistance(current[1][0],parking[1][0],current[1][1],parking[1][1])
            distances[2] = getDistance(current[2][0],parking[2][0],current[2][1],parking[2][1])
            distances[3] = getDistance(current[3][0],parking[3][0],current[3][1],parking[3][1])
        else:
            distances[1] = getDistance(current[0][0],parking[2][0],current[0][1],parking[2][1])
            distances[0] = getDistance(current[1][0],parking[3][0],current[1][1],parking[3][1])
            distances[3] = getDistance(current[2][0],parking[0][0],current[2][1],parking[0][1])
            distances[2] = getDistance(current[3][0],parking[1][0],current[3][1],parking[1][1])
        return distances
def getObrs(current,parking,parkingNumber,refVector,newVector,frontRight,backRight,x,y,mappingOffset):
    Bottom = True
    if parkingNumber%2 == 0:
        i = parkingNumber/2
    else:
        i = (parkingNumber-1)/2
        Bottom = False
    if i!= 0:
        i += ((i-1)/3)
    parkingOffset = -135
    const_bias = 70 #70
    distances = getDistances(current,parking,Bottom)
    heading = calcAngle(refVector,newVector,frontRight,backRight)
    distances = np.array(distances, dtype=float)
    heading = np.array([heading], dtype=float)
    overAllOffset = [None,None,None]
    overAllOffset[0] = mappingOffset[0]+const_bias+(i*parkingOffset)
    overAllOffset[1] = mappingOffset[1]
    if Bottom:
        pos = np.array([x-470+overAllOffset[0],y-240+overAllOffset[1]], dtype=float)
        final = {'Distances':distances,'Heading': heading,'Position':pos}
        overAllOffset[2] = 1
    else:
        pos = np.array([x-470+overAllOffset[0],y-240+overAllOffset[1]], dtype=float)
        pos[1] = pos[1]*-1
        final = {'Distances':distances,'Heading': -heading,'Position':pos}
        overAllOffset[2] = -1
    #print(final['Position'])
    return final,overAllOffset