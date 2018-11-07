# Project 3 - Playing Tennis
The task is a cooperative task because the aim is to keep
the ball for a long time in the air to reach the requested average reward of
0.5. So it is suitable to apply the same agent to both tennis players.
That's why I used a simple DDPG [1] agent to solve the task. 
The agent learns from the experiences from the replay buffer. 
The replay buffer collects the experiences of both agents.


## Hyperparameters
The following hyperparameters are used.
```

BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 128  
GAMMA = 0.99  
TAU = 0.2  
LR_ACTOR = 0.0001  
LR_CRITIC = 0.001  
WEIGHT_DECAY = 0.0  

```

## Architecture of Actor Network
```
Input Layer: 33 Neurons
Linear Layer 1: 64 Neurons, ReLu
Linear Layer 2: 64 Neurons, ReLu
Output Layer: 4 Neurons, Tanh
```

## Architecture of Critic Network
```
Input Layer1: 33 Neurons
Linear Layer 1: 64 Neurons, ReLU
Input Layer 2: 64 + 4 Neurons
Linear Layer 2: 64 Neurons, ReLu
Output Layer: 4 Neurons, Tanh
```

## Results
  * Environment solved in 989 episodes
  ![rewards](figures/reward_plot.png)
  
## Future Work
As a variant a real multi-agent approach like MADDPG [2] could be applied.
I doesn't seem to be necessary and might be overkill. It would be interesting 
to compare the results and learning speed and see if [2] brings any benefit.

  
  
## References
  *  [1] DDPG-algorithm: CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING, Lillicrap et. al., 2016
  *  [2] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, Lowe et. al, 2018