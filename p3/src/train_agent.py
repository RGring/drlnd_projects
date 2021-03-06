from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
import torch
import matplotlib.pyplot as plt
from collections import deque
import random

seed = 1

path_to_env = "../../envs/p3_collab-compet/Tennis_Linux/Tennis.x86_64"
path_to_save_model = "../saved_models"

env = UnityEnvironment(file_name=path_to_env,no_graphics=True)

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


#################################
######## AGENT TRAINING #########
#################################
agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, seed=seed)
def ddpg(n_episodes=2000, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations  # get the current state
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            # get next_states, rewards, dones for each agent.
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break
        scores_deque.append(np.amax(score))
        scores.append(np.amax(score))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), np.mean(score)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if (np.mean(scores_deque) > 0.5):
            torch.save(agent.actor_local.state_dict(), '%s/actor1_%d.pth' % (path_to_save_model, seed))
            torch.save(agent.critic_local.state_dict(), '%s/critic1_%d.pth' % (path_to_save_model, seed))
            break
    torch.save(agent.actor_local.state_dict(), '%s/actor1_%d.pth' % (path_to_save_model, seed))
    torch.save(agent.critic_local.state_dict(), '%s/critic1_%d.pth' % (path_to_save_model, seed))
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('Seed: %d' % seed)
plt.savefig('../figures/res1_%d.png' % (seed))

env.close()