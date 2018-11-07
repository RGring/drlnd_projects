# Project Environment Description

The environment simulates a Tennis match. It provides two rackets, that play
tennis agianst each other. They can execute two different continuous actions.
* moving forward/backwards
* jumping

The aim is to bring the ball over the net. It results in a reward of +0.1.
If the ball hits the ground or the net, the agent receives a reward of -0.01.

The observation space consists of 33 variables
corresponding to the position and velocity of the ball and racket.
The task is considered to be solved, if a score of 0.5 or more over 100 consecutive episodes.
As score the maximum of both agent should be considered.

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
  * [1] CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING, Lillicrap et. al.m 2016
