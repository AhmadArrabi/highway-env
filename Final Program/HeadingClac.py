import numpy as np
import numpy.linalg as LA

def getRightAngle(deg,point1,point2):
    if point1[1] < point2[1]:
        return deg*-1
    else:
        return deg

def calcAngle(a,b,point1,point2):
    a = np.array(a)
    #print(f"{a}     {b}")
    b = np.array(b)
    inner = np.inner(a, b)
    norms = LA.norm(a) * LA.norm(b)
    cos = inner / norms
    rad = np.arccos(cos)
    deg = np.rad2deg(rad)
    deg = getRightAngle(deg,point1,point2)
    return deg