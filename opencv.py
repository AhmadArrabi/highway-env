# 480 640 3
import random

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from numpy.core.numeric import outer
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
  
def preProcessing(input: np.ndarray):
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    output = Image.fromarray(edged)

    cv2.imshow("preprocessed",edged)
    print("Number of contours: ", str(len(cnts)))
    cv2.drawContours(input, cnts, 0, (0,255,0), 1)
    cv2.imshow("contours",input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    path = "D:\\University\\Graduation Project\\documentation\\figures\\parkings"
    fullpath = os.path.join(path, 'preprocessed2' + '.' + '.png')
    output.save(fullpath)


# define a video capture object
vid = cv2.VideoCapture(1)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(np.shape(frame))
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

preProcessing(frame)

