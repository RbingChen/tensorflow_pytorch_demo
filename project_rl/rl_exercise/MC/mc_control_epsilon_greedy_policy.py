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

def make_epsilon_greedy_policy(Q,epsilon,num_of_action):
    """
    根据给定的Q函数和epsilon创建一个epsilon-贪婪策略。
    参数：
    Q：一个从状态映射到动作-价值的字典。
    每个值都是一个长度为nA（见下文）的numpy数组
    epsilon：选择随机动作的概率。0到1之间的浮点数。
    num_of_action：环境中的动作数量。

    返回值：
    一个函数，它以观察为参数并返回
    每个动作的概率，以长度为nA的numpy数组的形式。
    """
    def policy_fn(observation):
        A = np.ones(num_of_action,dtype=float)*epsilon/num_of_action
        best_action = np.argmax(Q[observation])
        A[best_action] +=(1.0 - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    使用Epsilon-Greedy策略的蒙特卡洛控制。
    找到一个最优的epsilon-greedy策略。

    参数：
        env：OpenAI健身房环境。
        num_episodes：要采样的剧集数量。
        discount_factor：伽马折扣因子。
        epsilon：采样一个随机动作的机会。0到1之间的浮点数。

    返回值：
        一个元组(Q, 策略)。
        Q是一个将状态映射到动作值的字典。
        策略是一个函数，它以观察为参数并返回
        动作概率
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(lambda : np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q,epsilon,env.action_space.n)

    for eps in range(1, num_episodes + 1):
        if eps % 1000 == 0:
            print("\rEpisode {}/{}.".format(eps, num_episodes), end="")
            sys.stdout.flush()
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        print("episode")
        print(episode)
        states_in_episode = set([(tuple(x[0]),x[1]) for x in episode])
        print("states_in_episode")
        print(states_in_episode)
        for state,action in states_in_episode:
            sa_pair = (state,action)
            print("ss pair")
            print(sa_pair)
            first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state and x[1]==action)
            #print(first_occurence_idx)
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        print(returns_sum)
    return Q,policy


Q, policy = mc_control_epsilon_greedy(env, num_episodes=50, epsilon=0.1)
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
    print(state)
    print(actions)