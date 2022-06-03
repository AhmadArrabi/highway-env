import gym
import random
from sqlalchemy import false
import stable_baselines3
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
from numpy import array, inf
from numpy.core.fromnumeric import shape
import gym

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

env_name = "costume-parking-v0"
env = gym.make(env_name)

"""PPO_carDistance_noLineCrash - First succesfull model
PPO_carDistance_LineCrash - getting away from parking
PPO_carDistance_LinePenalty - started good but converged to not move
PPO_carDistance_LinePenalty2 - good baseline with episode = 1000, 100k training
"""

log_path = os.path.join("training", "logs")

model_name = "Task_1_100_freq" 
model_path = os.path.join("training", "saved_models", model_name)

model_name_2 = "Optimal_task2" 
model_path_2 = os.path.join("training", "saved_models", model_name_2)

model_name_3 = "Task_3_50_freq_2" 
model_path_3 = os.path.join("training", "saved_models", model_name_3)

model_name_booz = "Task1_GOAL" 
model_path_booz = os.path.join("training", "saved_models", model_name_booz)

print(check_env(env))

#model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path, gamma=0.75)#MlpPolicy
#model = PPO.load(f"{model_path_booz}/99000", env=env) #add-1.5mil #650000 peak1 940000 peak2 = 2.

Task_1 = PPO.load(f"{model_path_booz}/99000", env=env) #add-1.5mil #650000 peak1 940000 peak2 = 2.
Task_2 = PPO.load(f"{model_path_2}/99000", env=env) #add-1.5mil #650000 peak1 940000 peak2 = 2.
#Task_3 = PPO.load(f"{model_path_3}/145000", env=env) #add-1.5mil #650000 peak1 940000 peak2 = 2.

TIMESTEPS = 1000

#for i in range(1, 100):
#    model.learn(total_timesteps= TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_30")
#    model.save(f"{model_path_booz}/{TIMESTEPS*i}") 

done = False
obs = env.reset()
action, _states = Task_1.predict(obs)
#env.render()
do3 = False
do2 = False

while not done:
    #action = 1
    #action, _states = model.predict(obs)
    env.render()
    obs, rewards, done, info = env.step(action)

    #if (np.abs(obs['Heading']) < 92) & (np.abs(obs['Heading']) > 88) or do3:# & (np.abs(obs['Position'][0] - -165)<5) :
    #    do3=True
    #    print('TASK 3')
    #    action, _states = Task_3.predict(obs)
    if ((np.abs(obs['Heading']) > 20) and (obs['Position'][0]>-95))or do2:
        do2=True
        print('TASK 2')
        action, _states = Task_2.predict(obs)
    else:  
        print('TASK 1')
        action, _states = Task_1.predict(obs)

    """
    #TOP PARKING MAPPING
    if action == 0:
        action = 18
    elif action == 1:
        action = 19
    elif action == 2:
        action = 20
    elif action == 3:
        action = 15
    elif action == 4:
        action = 16
    elif action == 5:
        action = 17
    elif action == 6:
        action = 12
    elif action == 7:
        action = 13
    elif action == 8:
        action = 14
    elif action == 12:
        action = 6
    elif action == 13:
        action = 7
    elif action == 14:
        action = 8
    elif action == 15:
        action = 3
    elif action == 16:
        action = 4
    elif action == 17:
        action = 5
    elif action == 18:
        action = 0
    elif action == 19:
        action = 1
    elif action == 20:
        action = 2
    """


#Action:  0         Speed:  -10.0        Angle:  -30
#Action:  1         Speed:  0.0           Angle:  -30
#Action:  2         Speed:  10.0         Angle:  -30
#Action:  3         Speed:  -10.0        Angle:  -20
#Action:  4         Speed:  0.0           Angle:  -20
#Action:  5         Speed:  10.0         Angle:  -20
#Action:  6         Speed:  -10.0        Angle:  -10
#Action:  7         Speed:  0.0            Angle:  -10
#Action:  8         Speed:  10.0          Angle:  -10
#
#Action:  12       Speed:  -10.0      Angle:  10
#Action:  13       Speed:  0.0          Angle:  10
#Action:  14       Speed:  10.0        Angle:  10
#Action:  15       Speed:  -10.0       Angle:  20
#Action:  16       Speed:  0.0          Angle:  20
#Action:  17       Speed:  10.0        Angle:  20
#Action:  18       Speed:  -10.0       Angle:  30
#Action:  19       Speed:  0.0           Angle:  30
#Action:  20       Speed:  10.0         Angle:  30

    #action = 1
        
    print(info, rewards)
