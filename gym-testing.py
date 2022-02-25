import gym
import random
import highway_env
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
#it's hamza
from numpy import array, inf
from numpy.core.fromnumeric import shape
#
##funnctions
#PreProcessing function, takes a rendered image and return the state of the agent (distances)
def preProcessing(input: np.ndarray):
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #output = Image.fromarray(edged)
    ##print(shape(cnts))
    
    ##print("Number of contours: ", str(len(cnts)))
    cv2.drawContours(input, cnts, 0, (0,255,0), 1)
    return edged
    ##print(cnts[0], "####")
    ##print(cnts[0][0], "####")
    ##print(cnts[0][0][0],"#####")
    ##print(cnts[0][0][0][0],"#####")

    ##distanceMin = 99999999
    ##for xA, yA in cnts[0]:
    ##    for xB, yB in cnts[1]:
    ##        distance = ((xB-xA)**2+(yB-yA)**2)**(1/2) # distance formula
    ##        if (distance < distanceMin):
    ##            distanceMin = distance
    ##print(distanceMin)

    #cv2.imshow("contours",input)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#
#    #path = "D:\\University\\Graduation Project\\documentation\\figures\\parkings"
#    #fullpath = os.path.join(path, 'preprocessed' + '.' + '.png')
#    #output.save(fullpath)
#
#
##
#
#def preProcessing2(input):  
#    #input = input[1:1280,155:1500]  
#    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
#    blur = cv2.GaussianBlur(gray, (7,7), 0)
#    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 100)
#
#    # Dilate to combine adjacent text contours
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
#    dilate = cv2.dilate(thresh, kernel, iterations=3)
#    s=dilate
#    # Find contours, highlight text areas, and extract ROIs
#    cnts = cv2.findContours(s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#    if(len(cnts)):
#        line_items_coordinates = []
#        for c in cnts:
#            area = cv2.contourArea(c)
#            x,y,w,h = cv2.boundingRect(c)
#            image = cv2.rectangle(input, (x,y), (x+w, y+h), color=(0,255,0), thickness=2)
#            line_items_coordinates.append([(x,y), (x+w, y+h)])
#        return image
#    else:
#        return input
#
#env setup
env_name = "costume-parking-v0"
env = gym.make(env_name)
env.configure({
    "action": {
        "type": "DiscreteAction"
    }
})
#env = gym.wrappers.Monitor(env, force=True, directory="run", video_callable=lambda e: True) 

# record all episodes
# Feed the monitor to the wrapped environment, so it has access to the video recorder
# and can send it intermediate simulation frames.
#env.unwrapped.set_monitor(env)
env.reset()
done = False
count = 0


for i in range(10): 
    env.step(19)
    env.render()

while not done:
    count += 1
    #env.step(env.action_space.sample())  # with manual control, these actions are ignored
    x = env.render('rgb_array')
    #img = np.array(Image.fromarray(x))
    #preprocessed = preProcessing(img)
    #19 straight
    #15 backwards
    #24 speed 20 angle 10
    #23 speed 10 angle 10
    action = 0  #env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    print(count, info)


#cv2.imshow('image', img)
#cv2.imshow('preprocessed', preprocessed)
#cv2.waitKey(0)
#cv2.destroyAllWindows()    

#path = "D:\\University\\Graduation Project\\documentation\\figures\\parkings"   
#fullpath = os.path.join(path, '2D' + '.' + '.png')
#img = Image.fromarray(img)
#img.save(fullpath)



#plt.imshow(x)

   
#print(shape(img), type(img))

#preProcessing(img)

#-------for manual driving------#
#env.configure({
#    "manual_control": True
#})
#env.reset()
#done = False
#while not done:
#    env.step(env.action_space.sample())  # with manual control, these actions are ignored
#    env.render()
#RL standard loop
#c=0
#while True:
#    c+=1
#    print(c)
#    action = env.action_space.sample()
#    obs, rewards, done, info = env.step(action)
#    env.render()
#    print(info)
#
#    if done:
#        break