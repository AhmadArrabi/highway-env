import numpy as np
def goodFrame(results):
    perfectA = 7500
    minA = 2000
    maxA = 10000
    temp = [None]*len(results)
    print(f"len results = {len(results)}")
    i = 0
    for result in results:
        box = result[1][1]
        area = box[0] * box[1]
        #print(f"car area = {area}")
        if area >= minA and area <= maxA:
            diff = np.abs(perfectA - area)
        else:
            diff = float("inf")
        temp[i] = diff
        #print(f"{diff}   ",end="")
        i+=1
    print(f"dif = {temp}")
    # print(f"min dif = {min(temp)}")
    if not (min(temp) == float('inf')):
        ourCarIdx = temp.index(min(temp))
        #print(f"ourCarIdx = {ourCarIdx}")
        ourCar = results[ourCarIdx]
        otherObjects = []
        for k in range(len(results)):
            if k != ourCarIdx:
                otherObjects.append(results[k])
    else:
        ourCar = []
        otherObjects = []
#    print(f"len others = {len(otherObjects)}")
    # print(f"len car = {len(ourCar)}")
    return ourCar, otherObjects