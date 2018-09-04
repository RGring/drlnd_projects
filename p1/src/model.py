import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, dueling = False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling

        if self.dueling:
            a1_size = 32
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)

            self.v1 = nn.Linear(fc2_units, 1)

            self.a1 = nn.Linear(fc2_units, a1_size)
            self.a2 = nn.Linear(a1_size, action_size)

        else:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        if self.dueling:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            val = self.v1(x)
            adv = F.relu(self.a1(x))
            adv = self.a2(adv)
            return val + (adv - adv.mean())
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
