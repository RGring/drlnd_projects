# Project Environment Description
There is a simple finite world with blue and yellow bananas. An agent is able to move on the floor and collect the bananas by passing through them. The aim is to collect as much yellow bananas as possible, but leave the blue ones at the same time. For collecting a yellow banana the agents gets a reward of +1, while collecting a blue banana results in a reward of -1.

The agent can take four discrete actions:
* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

The state space has a size of 37 dimensions.

The task is considered solved, if the agents reaches a score of 13 or more over 100 consecutive episodes.

# Installations
1. Install Python 3
2. Install the following python packages:
  * numpy
  * unityagents
  * collections
  * torch
  * matplotlib
3. Download the environment and unpack the zip-folder.
  * Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  * Mac: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  * Windows 32-bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  * Windows 64-bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

# Getting Started
## How to train an agent
For training an agent start train_agent.py with python3 as interpreter. Please set the following variables in the script.
  * path_to_env: The path to the simulation environment.
  * path_to_save_model: The path, where the trained network model should be saved to.
  * Feel free to modify any hyperparamters.

## How to apply a trained agent
For running an already trained agent start the run_agent.py script with python3 interpreter. Pleases set the following variables in the scrip.
  * path_to_env: The path to the simulation environment.
  * path_to_save_agent: The path, where the network parameter should be loaded from.
  * dueling: If you load a dueling network, it need to be set to true and vice versa.
