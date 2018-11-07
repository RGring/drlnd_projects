import tensorflow as tf

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            fc1_units = 64
            fc2_units = 64

            """
            First layer:
            DENSE
            RELU
            """

            self.fc1 = tf.layers.Dense(inputs=state_size, units=fc1_units, activation=tf.nn.relu)

            """
            Second layer:
            DENSE
            RELU
            """
            self.fc2 = tf.layers.Dense(inputs=self.fc1, units = fc2_units, activation=tf.nn.relu)

            """
            Third layer:
            DENSE
            ELU
            """
            self.output = tf.layers.Dense(inputs=self.fc2, units = action_size, activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

