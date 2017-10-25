#
# Mountaincar problem using DQN and seperate target network
#

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque


class DQN:
    def __init__(self, env):
        # Environment to use
        self.env = env
        # Replay memory
        self.memory = deque(maxlen=10000)

        # Discount factor
        self.gamma = 0.99

        # Initial exploration factor
        self.epsilon = 1.0
        # Minimum value exploration factor
        self.epsilon_min = 0.005
        # Decay for epsilon
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 50000

        self.batch_size = 64
        self.train_start = 1000
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Learning rate
        self.learning_rate = 0.001

        # Model being trained
        self.model = self.create_model()
        # Target model used to predict Q(S,A)
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(
            32, input_dim=self.state_size, activation='relu', kernel_initializer="he_uniform"))
        model.add(Dense(16, activation='relu', kernel_initializer="he_uniform"))
        model.add(Dense(self.env.action_space.n, activation="linear",
                        kernel_initializer="he_uniform"))
        model.compile(
            loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        # Decay exploration rate by epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size))
        update_target = np.zeros((self.batch_size, self.action_size))

        for i in range(self.batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                    np.amax(self.target_model.predict(next_state)[0])
            update_input[i] = state
            update_target[i] = target

        self.model.fit(update_input, update_target,
                       batch_size=self.batch_size, epochs=1, verbose=0)

    def target_train(self):
        # Simply copy the weights of the model to target_model
        self.target_model.set_weights(self.model.get_weights())
        return

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env = gym.make("MountainCar-v0")

    trials = 4000
    trial_len = 500

    dqn_agent = DQN(env=env)
    for trial in range(trials):
        cur_state = env.reset().reshape(1, 2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()

            cur_state = new_state
            if done:
                env.reset()
                dqn_agent.target_train()
                break

        print("Iteration: {} Score: -{}".format(trial, step))


if __name__ == "__main__":
    main()
