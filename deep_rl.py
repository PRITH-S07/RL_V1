# -*- coding: utf-8 -*-
"""Deep_RL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12wAODl4OjwqacXWO5blA4HPk71kkIIBn

## Install Dependencies
"""

!pip install gym
!pip install keras-rl2
!pip install tensorflow==2.3.0
!pip install keras

"""## Test Random Environment with OpenAI Gym"""

import gym
import random

env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

actions

episodes=10
for episode in range(1,episodes+1):
  states=env.reset()
  done=False
  score=0

  while not done:
    #env.render()
    action=random.choice([0,1])
    n_state,reward,done,info = env.step(action)
    score+=reward
  print('Episode:{} Score:{}'.format(episode,score))

"""## Create a Deep Learning Model with Keras"""

# Ideally we want around 200. Our DL model will learn the best action to take in that environment in order
# to maximize our score

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def build_model(states, actions):
  model = Sequential()
  model.add(Flatten(input_shape=(1,states)))
  model.add(Dense(24,activation='relu'))
  model.add(Dense(24,activation='relu'))
  model.add(Dense(actions,activation='linear'))
  return model

model = build_model(states,actions)
model.summary()

"""## Build Agent with Keras-RL"""

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy # policy based rl
from rl.memory import SequentialMemory # to maintain memory for our DQN agent

def build_agent(model, actions):
  policy = BoltzmannQPolicy()
  memory = SequentialMemory(limit=50000,window_length=1)
  dqn = DQNAgent(model=model,memory=memory,policy=policy,
                 nb_actions=actions,nb_steps_warmup=10,target_model_update=1e-2)
  return dqn

dqn = build_agent(model,actions)
dqn.compile(Adam(lr=1e-3),metrics=['mae'])
dqn.fit(env,nb_steps=50000,visualize=False,verbose=1)

scores=dqn.test(env,nb_episodes=100,visualize=False)
#nb_episodes=no. of games
print(np.mean(scores.history['episode_reward']))

#_ = dqn.test(env,nb_episodes=15,visualize=True)

"""## Reloading Agent from Memory"""

dqn.save_weights('dqn_weights.h5f',overwrite=True)

del model

del dqn
del env

env = gym.make('CartPole-v0')
actions = env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states,actions)
dqn = build_agent(model,actions)
dqn.compile(Adam(lr=1e-3),metrics=['mae'])

dqn.load_weights('dqn_weights.h5f')

scores=dqn.test(env,nb_episodes=100,visualize=False)
#nb_episodes=no. of games
print(np.mean(scores.history['episode_reward']))

#_ = dqn.test(env,nb_episodes=15,visualize=True)

