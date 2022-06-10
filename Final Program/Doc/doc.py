from transform import four_point_transform
from processImage import *
import cv2 as cv
import numpy as np
img = cv.imread("./Doc/image_resized_without_car.png")

pts = np.array([(79,2),(878,1),(49,539),(894,538)], dtype = "float32")
imgAfter = four_point_transform(img,pts)
imgAfter2 = cv.resize(imgAfter,(640,480))

frame = cv.imread("./Doc/image_resized.png")
imgAfter = four_point_transform(frame,pts)
imgAfter = cv.resize(imgAfter,(640,480))

sub = cv.subtract(imgAfter,imgAfter2)
c = processing(sub,20,7,False)
res = processing_Conts(c,[],-1)
cv.drawContours(imgAfter,[res[0][0]],0,(0,255,255),2)
cv.imshow("Car detected", imgAfter)
cv.waitKey(0)
cv.destroyAllWindows()

# cv.imshow('The reference image before transformation',img)
# cv.imshow('The reference image after transformation',imgAfter)