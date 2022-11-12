from getDistance import getDistance
from scipy.spatial import distance as dist
import numpy as np
# def setPoint(oldcords,newcords):
#     old_x = oldcords[0]
#     old_y = oldcords[1]
#     minDis = float("inf")
#     minIdx = -1
#     c = 0
#     finalOption = ()
#     for option in newcords:
#         new_x = option[0]
#         new_y = option[1]
#         distance = getDistance(old_x,new_x,old_y,new_y)
#         if distance < minDis:
#             if distance < 15:
#                 minIdx = c
#                 finalOption = (option[0],option[1])
#                 minDis = distance
#             else:
#                 finalOption = (old_x,old_y)
#         c+=1
#     newcords = np.delete(newcords,minIdx,axis=0)
#     #print(f"({old_x},{old_y}) -> {finalOption} dis:{minDis}")
#     return finalOption, newcords
def setPoints(old,new):
    finalPoints = [None]*len(old)
    oldPoints = np.array(old)
    newPoints = new
    D = dist.cdist(oldPoints, newPoints)
    D_Temp = D
    for i in range(len(D_Temp)):
        row = D_Temp[i]
        min_ = np.argmin(row)
        #print(f"{row}  {min_}  {row[min_]}")
        D_Temp[:, min_] = float("inf")
        finalPoints[i] = newPoints[min_]
    # print(oldPoints)
    # print("------")
    # print(newPoints)
    # print("++++++++")
    # print(D)
    # print(rows)
    # print(cols)
    # print(map)
    #print("##################################################################")
    return finalPoints
