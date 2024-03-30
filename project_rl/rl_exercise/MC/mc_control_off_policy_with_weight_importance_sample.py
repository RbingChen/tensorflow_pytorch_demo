# coding:utf-8

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")
from envs.blackjack import BlackjackEnv
from envs import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


def create_random_policy(num_of_action):
    A = np.ones(num_of_action, dtype=float) / num_of_action

    def policy_fn(observation):
        return A

    return policy_fn


def create_greedy_policy(Q):
    """
    """

    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A

    return policy_fn


def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    target_policy = create_greedy_policy(Q)

    for eps in range(1, num_episodes + 1):
        if eps % 1000 == 0:
            print("\rEpisode {}/{}.".format(eps, num_episodes), end="")
            sys.stdout.flush()
        episode = []
        state = env.reset()
        for t in range(100):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        G = 0.0
        W = 1.0
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount_factor * G + reward
            C[state][action] +=W
            Q[state][action] += (W/C[state][action])*(G-Q[state][action])

            if action != np.argmax(target_policy(state)):
                break
            W = W * 1.0/behavior_policy(state)[action]
    return Q, target_policy