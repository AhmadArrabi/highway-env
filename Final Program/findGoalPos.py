def findCenter(parkingPoints):
    x = []
    y = []
    for point in parkingPoints:
        x.append(point[0])
        y.append(point[1])
    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y)
    w = max_x - min_x
    h = max_y - min_y
    center = [(min_x+(w/2)) - 470,(min_y+(h/2))-240]
    return center

def findCenters(parkings):
    centers = []
    for parking in parkings:
        centers.append(findCenter(parking))
    return centers

def findXY_offset(goalPos1,parkingNum):
    goalSim = [[-165,177.5],[-165,-177.5],[-55,177.5],[-55,-177.5],[55,177.5],[55,-177.5]] #FIXED
    x = (goalSim[parkingNum][0] - goalPos1[0])
    y= (goalSim[parkingNum][1] - goalPos1[1])
    # x=-300
    # y=0
    mappingOffset = [x,y]
   # print(goalPos1,mappingOffset)
    return mappingOffset