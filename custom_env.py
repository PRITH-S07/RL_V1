## Import dependencies

# Import GYM stuff
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# Import helpers
import numpy as np
import random
import os

# Import Stable baselines stuff
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

## TYPES OF SPACES

print(Discrete(3).sample())
print(Box(0,1,shape=(3,)).sample()) # for sensors or continuous variables, it can be used
print(Tuple((Discrete(3),Box(0,1,shape=(3,)))).sample()) # allows us to combine diff. spaces
print(Dict({'height':Discrete(2),"speed":Box(0,100,shape=(1,))}).sample()) # just like tuple, except it's a dict
print(MultiBinary(4).sample()) # just diff. combos of 0s and 1s here
print(MultiDiscrete([5,2,2]).sample()) # 0->4, 0->1, 0->1

## BUILDING AN ENVT.
# Goal is to build an agent to give us the best shower possible
# Randomly, temp is to be decided so that the agent can localise the temp. between 37 and 39 deg.

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0,high=100,shape=(1,))
        self.state = 38+random.randint(-3,3)
        self.shower_length = 60
        pass
    def step(self, action):
        # Apply temp adj
        self.state+=action-1
        # Decrease shower time
        self.shower_length-=1
        # Calculate Reward
        if self.state>=37 and self.state<=39:
            reward = 1
        else:
            reward=-1
        if self.shower_length<=0:
            done=True
        else:
            done=False
        info={}
        return self.state,reward,done,info
    def render(self):
        # Implement viz
        pass
    def reset(self):
        self.state = np.array([38+random.randint(-3,3)]).astype(float)
        self.shower_length = 60
        return self.state
env = ShowerEnv()
print(env.observation_space.sample())
print(env.action_space.sample())

## TEST ENVT.
episodes=5
for episode in range(1,episodes+1):
  obs = env.reset()
  done = False
  score = 0
  while not done:
    env.render()
    action = env.action_space.sample()
    obs,reward,done,info = env.step(action)
    score+=reward
  print('Episode:{} Score:{}'.format(episode,score))
env.close()

## TRAIN MODEL

log_path='/Users/prithsharma/Desktop/Training/Logs'
model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

model.learn(total_timesteps=60000)

# SAVE MODEL
shower_path = '/Users/prithsharma/Desktop/Training/Saved Models/Shower_Model_PPO'
model.save(shower_path)

del model

model = PPO.load(shower_path,env)

print(evaluate_policy(model,env,n_eval_episodes=10,render=False))