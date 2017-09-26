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
        self.memory = deque(maxlen=2000)

        # Discount factor
        self.gamma = 0.85

        # Initial exploration factor
        self.epsilon = 1.0
        # Minimum value exploration factor
        self.epsilon_min = 0.01
        # Decay for epsilon
        self.epsilon_decay = 0.995

        # Learning rate
        self.learning_rate = 0.005

        # Model being trained
        self.model = self.create_model()
        # Target model used to predict Q(S,A)
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(
            loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        # Decay exploration rate by epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        # Simply copy the weights of the model to target_model
        self.target_model.set_weights(self.model.get_weights())
        return

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env = gym.make("MountainCar-v0")

    trials = 1000
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

            if step % 10 == 0:
                dqn_agent.target_train()

            cur_state = new_state
            if done:
                break

        print("Iteration: {} Score: {}".format(trial, step))


if __name__ == "__main__":
    main()
