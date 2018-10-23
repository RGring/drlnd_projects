# Project 1 - collecting tasty bananas
To solve the learning task the DDPG-algorithm is applied here. 


## Hyperparameters
The following hyperparameters are used.
```

BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 128  
GAMMA = 0.99  
TAU = 0.001  
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
  * Environment solved in ~150 episodes
  
  
## References
  *  DDPG-algorithm: CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING, Lillicrap et. al.m 2016
