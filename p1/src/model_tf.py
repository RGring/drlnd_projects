import tensorflow as tf

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
        self.seed = tf.set_random_seed(seed)
        self.dueling = dueling

        if self.dueling:
            a1_size = 32
            self.fc1 = tf.layers.Dense(inputs=state_size, units = fc1_units, activation=tf.nn.relu)
            self.fc2 = tf.layers.Dense(inputs=fc1_units, units = fc2_units, activation=tf.nn.relu)
            self.v1 = tf.layers.Dense(inputs=fc2_units, units=1)

            self.a1 = tf.layers.Dense(inputs=fc2_units, units=a1_size, activation=tf.nn.relu)
            self.a2 = tf.layers.Dense(inputs=a1_size, units=action_size)

        else:
            self.fc1 = tf.layers.Dense(inputs=state_size, units = fc1_units, activation=tf.nn.relu)
            self.fc2 = tf.layers.Dense(inputs=fc1_units, units = fc2_units, activation=tf.nn.relu)
            self.fc3 = tf.layers.Dense(inputs=fc2_units, units = action_size, activation=None)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        if self.dueling:
            x = self.fc1(state)
            x = self.fc2(x)
            val = self.v1(x)
            adv = self.a1(x)
            adv = self.a2(adv)
            return val + (adv - adv.mean())
        else:
            x = self.fc1(state)
            x = self.fc2(x)
            return self.fc3(x)
