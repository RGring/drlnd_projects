# Example code using opanAI gym and vrep for a q-learning agent based on work by Massimo Innocenti

from unityagents import UnityEnvironment
import tensorflow as tf
import numpy as np
import time


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, LSTMCell, GRUCell, GRU
from keras.optimizers import Adam

# These imports handle the RL Agents implemented in the matthiasplappert/keras-rl git repository
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import sys

if __name__ == '__main__':
    # Set Tensorflow to use only 15% of available memory per process
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # Set environment
    path_to_env = "/home/ronja/MiR/Udacity/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64"

    env = UnityEnvironment(file_name=path_to_env, no_graphics=True, seed=0)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    #################################
    ######## ENVIRONENT INFO ########
    #################################

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)


    # Next, we build a very simple model.
    fc1_units = 64
    fc2_units = 64

    model = Sequential()
    model.add(Dense(fc1_units, input_shape=(1,state_size), activation=tf.nn.relu))
    model.add(Dense(fc2_units, activation=tf.nn.relu))
    model.add(Dense(action_size))
    model.add(Activation('softmax'))

    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, verbose=2, action_repetition=1)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_weights.h5f',overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5)
