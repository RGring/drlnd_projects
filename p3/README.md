# Project Environment Description

The environment consists of an arm with two joints. Furthermore there is a moving blobb, that specifies the target region, that need to be reached
by the arm. For each timestep the arm is at that target region, it gets a reward of 0.1. The goal is to be at that position as many times as possible
to maximize the cummulative reward.

Each action is a vector with four numbers, corresponding to torque applicable to two joints. 
Every entry in the action vector should be a number between -1 and 1.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

The task is considered to be solved, if the agents reaches a score of 30 or more over 100 consecutive episodes.

There are two differents variants of the environment:
    * Env1 includes a single arm
    * Env2 includes 20 agents. It allows to collect more experiences in a shorter time.

# Installations
1. Install Python 3
2. Install the following python packages:
  * numpy
  * unityagents
  * collections
  * torch
  * matplotlib
  * collections
  * copy
3. Download the environment with 20 agents and unpack the zip-folder.
  * Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  * Mac: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  * Windows 32-bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  * Windows 64-bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

# Getting Started
## How to train an agent
For training an agent start train_agent.py with python3 as interpreter. Please set the following variables in the script.
  * path_to_env: The path to the simulation environment.
  * path_to_save_model: The path, where the trained network model should be saved to.
  * Feel free to modify any hyperparamters.

## How to apply a trained agent
For running an already trained agent start the run_agent.py script with python3 interpreter. Pleases set the following variables in the script.
  * path_to_env: The path to the simulation environment.
  * path_to_save_agent: The path, where the network parameter should be loaded from.

## References
  * Code structure is from the Udacity lesson coding exercise DDPG BiPedal
  * CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING, Lillicrap et. al.m 2016
