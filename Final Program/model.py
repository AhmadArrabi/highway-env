import numpy as np
from mapAction import mapAction
from mapAction import *
do3 = False
do2 = False
done = False
def getAction_Server(obs,Task_1,Task_2,Task_3,goalPos,overAllOffset):
    global do3,do2,done
    action =None
    _states = None
    task = None
    #goalPos = [-179,186] ########################################
    actualPos = obs['Position']
    actualPos = np.array([actualPos[0]-overAllOffset[0],(actualPos[1]-overAllOffset[1])*overAllOffset[2]])
    #print(mappingOffset)
    #print(f"{actualPos}     {goalPos}     {np.linalg.norm(goalPos - actualPos)}")
    #print(f"{goalPos[0]+470+mappingOffset[0]}   {goalPos[1] +240+mappingOffset[1]}")
    
    #if (np.linalg.norm(actualPos - goalPos) < 32): #40
    #print(np.abs(goalPos[1] - actualPos[1]))
    if (np.abs(goalPos[1] - actualPos[1]) < 10): #18
        print("DOOOOOONE")
        action = 10
        done = True
    else:
        if (np.abs(obs['Heading']) < 90.5) & (np.abs(obs['Heading']) > 88.5) or do3: #and (np.abs(obs['Position'][0] - goalPos[0]) < 5
            do3 = True
            # action_ = 9
            #obs['Heading'][0] = -obs['Heading'][0]
            action_, _states = Task_3.predict(obs)
            #action = mapAction2(action_)
            action = action_
            print("TASK 3")
            task = 3
        elif (np.abs(obs['Heading']) > 30) or do2: #40
            do2 = True
            action, _states = Task_2.predict(obs)
            print("TASK 2")
            task = 2
        else:
            obs_new = obs
            # obs_new['Heading'] = -obs_new['Heading']
            # obs_new['Position'][1] = -obs_new['Position'][1]
            action, _states = Task_1.predict(obs_new)
            # action = mapAction(action)
            print("TASK 1")
            task = 1
    return action,done,task