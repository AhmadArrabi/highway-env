def getXY(array):
    x = []
    y = []
    for point in array:
        x.append(point[0])
        y.append(point[1])
    startX = min(x)
    endX = max(x)
    startY = min(y)
    endY = max(y)
    return (startX,startY,endX,endY)