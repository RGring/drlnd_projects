from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt


from dqn_agent import Agent

path_to_env = "/home/ronja/MiR/Udacity/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64"
path_to_model = "../saved_models/dueling.pth"
dueling = True

env = UnityEnvironment(file_name=path_to_env,no_graphics=False, seed = 0)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#################################
######## ENVIRONENT INFO ########
#################################

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

##################################
########### LOAD AGENT ###########
##################################

agent = Agent(state_size=state_size, action_size=action_size, dueling=dueling, seed=0)
agent.load_model(path_to_model)

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score
while True:
    action = agent.act(state, 0)  # select an action
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    score += reward  # update the score
    state = next_state  # roll over the state to next time step
    if done:  # exit loop if episode finished
        break

print("Score: {}".format(score))
env.close()
