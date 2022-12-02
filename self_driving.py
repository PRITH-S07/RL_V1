#!pip install gym[box2d] pyglet
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name='CarRacing-v0'
env = gym.make(environment_name)
env.reset()
print(env.action_space)
print(env.observation_space)

## sample viewing
for i in range(1000):
    env.step(env.action_space.sample())
    env.render() # gives the ability to see the agent in action

## TESTING THE ENVT:
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

## TRAINING THE MODEL:
env = gym.make(environment_name)
env = DummyVecEnv([lambda:env])

log_path = '/Users/prithsharma/Desktop/Training/Logs'
model = PPO('CnnPolicy',env,verbose=1,tensorboard_log=log_path)

## 900 score for training is good, however will take many iterations and v long

model.learn(total_timesteps=100000)

#saving the model
ppo_path = '/Users/prithsharma/Desktop/Training/Saved Models/PPO_Driving_Model'

model.save(ppo_path)

del model

#reloading model
ppo_path = '/Users/prithsharma/Desktop/Training/Saved Models/PPO_Driving_Model'
model = PPO.load(ppo_path,env)

#EVALUATE AND TEST
#evaluate_policy(model,env,n_eval_episodes=10,render=True)

## TESTING
episodes=5
for episode in range(1,episodes+1):
  obs=env.reset() # to get initial set of observations
  done=False
  score=0
  while not done:
    obs = obs.copy()
    env.render() # to view the representation of the envt..
    action,_=model.predict(obs) # now using the model here
    # env.observation_space results in a Box envt. output
    obs,reward,done,info=env.step(action)
    score+=reward
  print('Episode:{} Score:{}'.format(episode,score))
env.close()