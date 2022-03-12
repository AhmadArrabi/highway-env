from math import gamma
import gym
import random
from sqlalchemy import false
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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

env_name = "costume-parking-v0"
env = gym.make(env_name)

log_path = os.path.join("training", "logs")
model_name = "A2C_low2_gamma"
model_path = os.path.join("training", "saved_models", model_name)

#print(check_env(env))

#model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_path, gamma = 0.05)
#model.learn(total_timesteps=20000)
#model.save(model_path)

#del model # remove to demonstrate saving and loading

#model = A2C.load(model_path, env=env)

#model.learn(total_timesteps=30000)

#model.save(model_path)

done = False
obs = env.reset()

while not done:
    #action, _states = model.predict(obs)
    action = 19
    obs, rewards, done, info = env.step(action)
    env.render()
    print(info)

#for i in range(20): 
#    env.step(19)
#    env.render()
#
#count =0
#done = False
#while not done:
#    count += 1
#        #env.step(env.action_space.sample())  # with manual control, these actions are ignored
#    x = env.render('rgb_array')
#        #img = np.array(Image.fromarray(x))
#        #preprocessed = preProcessing(img)
#        #19 straight
#        #15 backwards
#        #24 speed 20 angle 10
#        #23 speed 10 angle 10
#    action = 24 #env.action_space.sample()
#    obs, rewards, done, info = env.step(action)
#    print(count, rewards, info)
#
