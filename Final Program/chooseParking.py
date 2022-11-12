from scipy.spatial import distance as dist
import numpy as np
from getDistance import getDistance
def chooseParking(full,result,goals):
    av_goals = []
    for par in full.keys():
        print(f"# = {par}      st = {full[par]}")
        if full[par]==True:
            av_goals.append(par)
    print(av_goals)
    center = np.array([result[1][0][0]-470,result[1][0][1]-240])
    minD = float("inf")
    minIdx = -1
    for par in av_goals:
        parking = np.array(goals[par])
        D = getDistance(center[0],parking[0],center[1],parking[1])
        # print(parking)
        # print(center)
        if D < minD:
            minD = D
            minIdx = par
    return minIdx
