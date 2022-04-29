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

#model_name = "Task_1_100_freq" 
#model_path = os.path.join("training", "saved_models", model_name)

model_name_2 = "Task_2_50_freq" 
model_path_2 = os.path.join("training", "saved_models", model_name_2)

#model_name_3 = "Task_3_50_freq_2" 
#model_path_3 = os.path.join("training", "saved_models", model_name_3)

print(check_env(env))

#model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)#MlpPolicy
model = PPO.load(f"{model_path_2}/Second_training_245000", env=env) #add-1.5mil #650000 peak1 940000 peak2 = 2.

#Task_1 = PPO.load(f"{model_path}/195000", env=env) #add-1.5mil #650000 peak1 940000 peak2 = 2.
#Task_2 = PPO.load(f"{model_path_2}/295000", env=env) #add-1.5mil #650000 peak1 940000 peak2 = 2.
#Task_3 = PPO.load(f"{model_path_3}/145000", env=env) #add-1.5mil #650000 peak1 940000 peak2 = 2.

TIMESTEPS = 5000

#for i in range(1, 50):
#    model.learn(total_timesteps= TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_23")
#    model.save(f"{model_path_2}/Second_training_{TIMESTEPS*i}") 

done = False
obs = env.reset()
#action, _states = Task_1.predict(obs)
#env.render()

while not done:
    #action = 9
    action, _states = model.predict(obs)
    env.render()
    obs, rewards, done, info = env.step(action)

    #if (np.abs(obs['Heading']) < 93) & (np.abs(obs['Heading']) > 87):
    #    action, _states = Task_3.predict(obs)
    #else: action, _states = model.predict(obs)
    
    #if (np.abs(obs['Heading']) < 92) & (np.abs(obs['Heading']) > 88):
    #    action, _states = Task_3.predict(obs)
    #elif (np.abs(obs['Heading']) > 30):
    #    action, _states = Task_2.predict(obs)
    #else: action, _states = Task_1.predict(obs)

    print(info, rewards)
