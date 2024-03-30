# coding:utf-8

import numpy as np
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.toy_text.utils import categorical_sample


class DiscreteEnv(Env):
    """
    num_of_state:状态数量
    num_of_action：动作熟练
    transition_prob：转移概率矩阵 ,P[s][a] = [[概率,下个状态，reward，done),.....]
    init_state_distribution：初始化状态分布
    尤其注意 转移概率矩阵的
    """

    def __init__(self, num_of_state, num_of_action, transition_prob, init_state_distribution):
        self.num_of_state = num_of_state
        self.num_of_action = num_of_action
        self.transition_proba = transition_prob
        self.init_state_distribution = init_state_distribution
        self.last_action = None
        self.action_space = spaces.Discrete(self.num_of_action)
        self.state_space = spaces.Discrete(self.num_of_state)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = categorical_sample(self.init_state_distribution, self.np_random)
        self.last_action = None
        return int(self.state)

    def step(self, action):
        transitions = self.transition_proba[self.state][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, state, reward, done = transitions[i]
        self.state = state
        self.last_action = action
        return int(self.state), reward, done, {"prob": prob}
