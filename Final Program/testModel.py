from model import getAction_Server
import numpy as np
##################3
import os
from stable_baselines3 import PPO
model_name = "Task_1_100_freq" 
model_path = os.path.join("models", model_name)
model_name_2 = "Task_2_50_freq" 
model_path_2 = os.path.join("models", model_name_2)
model_name_3 = "Task_3_50_freq_2" 
model_path_3 = os.path.join("models", model_name_3)
Task_1 = PPO.load(f"{model_path}/195000")
Task_2 = PPO.load(f"{model_path_2}/Second_training_240000") 
Task_3 = PPO.load(f"{model_path_3}/145000")

obsrevs = {'Distances': np.array([50,50,50,50], dtype=float),
    'Heading': np.array([-90], dtype=float),
    'Position': np.array([0,-50], dtype=float)}
action = getAction_Server(obsrevs,Task_1,Task_2,Task_3)

print(action)