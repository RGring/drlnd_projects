from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt


from ddpg_agent import Agent


path_to_env = "/home/ronja/MiR/Udacity/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux_single/Reacher.x86_64"
path_to_env = "//home/ronja/MiR/Udacity/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64"

path_to_critic = "../saved_models/critic.pth"
path_to_actor = "../saved_models/actor.pth"

env = UnityEnvironment(file_name=path_to_env,no_graphics=False, seed = 0)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#################################
######## ENVIRONENT INFO ########
#################################

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

##################################
########### LOAD AGENT ###########
##################################

agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, seed=1)
agent.load_model(path_to_actor, path_to_critic)

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
states = env_info.vector_observations  # get the current state
score = np.zeros(num_agents)  # initialize the score
while True:
    actions = agent.act(states)  # select an action
    env_info = env.step(actions)[brain_name]  # send the action to the environment
    next_states = env_info.vector_observations  # get the next state
    rewards = env_info.rewards  # get the reward
    dones = env_info.local_done  # see if episode has finished
    score += rewards  # update the score
    states = next_states  # roll over the state to next time step
    if np.any(dones):
        break

print("Score: {}".format(score))
env.close()
