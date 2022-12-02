# -*- coding: utf-8 -*-
"""RL_TUTORIAL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16kNTpFM-IbwojdzSLpu4d3FC7OrLbkvC

## Import dependencies
"""

!python --version

!pip install stable_baselines3[extra]
!pip install gym[all]

import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv # To vectorize envts, to train ML model or RL agent on multiple envts at the same time...a huge boost in training speed
# Just a wrapper around our envt...makes it easier to work with stable baselines
from stable_baselines3.common.evaluation import evaluate_policy # to test how our model is performing

"""## Load the environment"""

environment_name='CartPole-v0'
env=gym.make(environment_name)

episodes=5
for episode in range(1,episodes+1):
  state=env.reset() # to get initial set of observations
  done=False
  score=0
  while not done:
    #env.render() # to view the representation of the envt..
    action=env.action_space.sample() # can do env.observation_space...action space here is Discrete(2)
    # env.observation_space results in a Box envt. output
    n_state,reward,done,info=env.step(action)
    score+=reward
  print('Episode:{} Score:{}'.format(episode,score))
#env.close()

"""## Understanding the environment"""

env.action_space

env.observation_space

env.action_space.sample()

env.observation_space.sample()

"""## Train an RL model"""

# make directories first
log_path = os.path.join('Training','Logs')

print(log_path)

env=gym.make(environment_name)
env=DummyVecEnv([lambda:env]) # wrapper for a non-vectorized env.
model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path) # here we are using PPO algorithm
# using an MLP policy here....just using standard neural network layers

model.learn(total_timesteps=20000)

"""## Save and Reload Model"""

PPO_Path = os.path.join('Training','Saved Models','PPO_model_cartpole')

model.save(PPO_Path)

del model

PPO_Path

model = PPO.load(PPO_Path,env=env)

"""## Evaluation"""

# PPO model is considered solved if on average we get a score of 200 or higher
evaluate_policy(model,env,n_eval_episodes=10,render=False)

env.close()

"""## Test Model"""

episodes=5
for episode in range(1,episodes+1):
  obs=env.reset() # to get initial set of observations
  done=False
  score=0
  while not done:
    #env.render() # to view the representation of the envt..
    action,_=model.predict(obs) # now using the model here
    # env.observation_space results in a Box envt. output
    obs,reward,done,info=env.step(action)
    score+=reward
  print('Episode:{} Score:{}'.format(episode,score))
#env.close()

env.close()

action, _ =model.predict(obs)
env.step(action)

"""## Viewing Logs in Tensorboard"""

training_log_path = '/content/Training/Logs/PPO_1'
training_log_path

!tensorboard --logdir={training_log_path}

"""## Adding a callback to the training Stage"""

from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnRewardThreshold

save_path=os.path.join('Training','Saved Models')

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200,verbose=1)
eval_callback=EvalCallback(env,
                           callback_on_new_best=stop_callback,
                           eval_freq=10000,
                           best_model_save_path=save_path,
                           verbose=1)

model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

model.learn(total_timesteps=20000,callback=eval_callback)

"""## Changing Policies"""

net_arch=[dict(pi=[128,128,128,128],vf=[128,128,128,128])] # new nn arch

# pi -> custom actor, new nn m 4layer with 128 units each layer, value function with 4 layer 128 unit per layer

model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path,policy_kwargs={'net_arch':net_arch})

model.learn(total_timesteps=20000,callback=eval_callback)

"""## Using an Alternate Algorithm"""

from stable_baselines3 import DQN

model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

model.learn(total_timesteps=20000)

# Type in DQN.load() after model.save() here.