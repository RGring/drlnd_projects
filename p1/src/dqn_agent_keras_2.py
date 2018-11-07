# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy.random
from tensorflow import set_random_seed

EPISODES = 1000
UPDATE_EVERY = 4  # how often to update the network


class QNetwork:
    def __init__(self, state_size, action_size, learning_rate, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.seed = seed
        numpy.random.seed(seed)
        set_random_seed(seed)
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))

    def getModel(self):
        self.model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return self.model


class Agent:
    def __init__(self, state_size, action_size, seed=0, dueling=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=int(1e5))
        self.gamma = 0.99    # discount rate
        self.learning_rate = 5e-4
        self.batch_size = 64
        self.seed = seed
        self.t_step = 0
        self.model = QNetwork(self.state_size, self.action_size, self.learning_rate, self.seed).getModel()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = random.sample(self.memory, self.batch_size)
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        if np.random.rand() <= eps:
            return random.randrange(self.action_size)
        act_values = self.model.predict(self.reshape_state(state))
        return np.argmax(act_values[0])  # returns action

    def learn(self, experiences, gamma):
        for state, action, reward, next_state, done in experiences:
            target = reward
            if not done:
                target = (reward + gamma *
                          np.amax(self.model.predict(self.reshape_state(next_state))[0]))
            target_f = self.model.predict(self.reshape_state(state))
            target_f[0][action] = target
            self.model.fit(self.reshape_state(state), target_f, epochs=1, verbose=0)

    def reshape_state(self, state):
        return np.reshape(state, [1, self.state_size])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# if __name__ == "__main__":
#     env = gym.make('CartPole-v1')
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     agent = DQNAgent(state_size, action_size)
#     # agent.load("./save/cartpole-dqn.h5")
#     done = False
#     batch_size = 32
#
#     for e in range(EPISODES):
#         state = env.reset()
#         state = np.reshape(state, [1, state_size])
#         for time in range(500):
#             # env.render()
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             reward = reward if not done else -10
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 print("episode: {}/{}, score: {}, e: {:.2}"
#                       .format(e, EPISODES, time, agent.epsilon))
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)
#         # if e % 10 == 0:
#         #     agent.save("./save/cartpole-dqn.h5")