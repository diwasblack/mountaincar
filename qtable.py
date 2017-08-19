"""
Q-Learning example using OpenAI gym MountainCar enviornment
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""

import numpy as np

import gym
from gym import wrappers

n_states = 50
eta = 0.03  # Learning rate
gamma = 0.95
learning_iters = 600000


def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    for _ in range(10000):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a, b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    return a, b


def get_random_policy(env):
    """ returns a random policy """
    return np.random.choice(env.action_space.n, size=(n_states, n_states))


if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    print('----- using Q Learning -----')
    q_table = np.zeros((n_states, n_states, 3))
    for i in range(learning_iters):
        obs = env.reset()
        total_reward = 0
        for j in range(10000):
            a, b = obs_to_state(env, obs)
            action = np.argmax(q_table[a][b])
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            # update q table
            a_, b_ = obs_to_state(env, obs)
            q_table[a][b][action] = q_table[a][b][action] + eta * (
                reward + gamma * np.max(q_table[a_][b_]) -
                q_table[a][b][action])
            if done:
                break
        if i % 100 == 0:
            print(i, ' - ', total_reward)
    policy = np.argmax(q_table, axis=2)
    eval_score = run_episode(env, policy, True)
    print("Evaluation score = ", eval_score)
    # exit(0)
    monitor_path = '/tmp/mountaincar_exp'
    env = wrappers.Monitor(env, monitor_path, force=True)
    for _ in range(2000):
        run_episode(env, policy)
    env.close()
    # gym.upload(monitor_path, api_key= ...)
