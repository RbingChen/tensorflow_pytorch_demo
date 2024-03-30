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


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    蒙特卡洛预测算法。使用采样计算给定策略的价值函数。

    参数：
        policy：一个将观察映射到动作概率的函数。
        env：OpenAI健身房环境。
        num_episodes：要采样的剧集数量。
        discount_factor：gamma因子。

    返回值：
        一个从状态映射到价值的字典。
        状态是一个元组，价值是一个浮点数。
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    for eps in range(1, num_episodes + 1):
        if eps % 1000 == 0:
            print("\rEpisode {}/{}.".format(eps, num_episodes), end="")
            sys.stdout.flush()
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        print("episode")
        print(episode)
        states_in_episode = set([tuple(x[0]) for x in episode])
        print("states_in_episode")
        print(states_in_episode)
        for state in states_in_episode:
            first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            print(first_occurence_idx)
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
    return V


def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


V_10k = mc_prediction(sample_policy, env, num_episodes=10)
#plotting.plot_value_function(V_10k, title="10,000 Steps")

# V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
# plotting.plot_value_function(V_500k, title="500,000 Steps")
