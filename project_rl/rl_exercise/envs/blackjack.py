# coding:utf-8
import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return np_random.choice(deck)


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """
    简单的二十一点环境
    二十一点是一种卡牌游戏，目标是获得总和尽可能接近21但不超过21的牌。玩家将与固定的庄家对战。
    面牌（杰克，皇后，国王）的点数为10。
    A牌可以计为11点或1点，当其为11点时，我们称之为'可用'。
    这个游戏是用无限牌库（或者说是可以替换的牌）来进行的。
    游戏开始时，每个人（玩家和庄家）都有一张面朝上和一张面朝下的牌。
    玩家可以请求额外的牌（命中=1），直到他们决定停止（停牌=0）或超过21点（爆牌）。
    在玩家停牌后，庄家揭示他们的面朝下的牌，并继续抽牌，直到他们的总和达到17点或更高。如果庄家爆牌，玩家赢。
    如果玩家和庄家都没有爆牌，那么结果（赢，输，平）将由谁的总和更接近21点来决定。赢的奖励是+1，平局是0，输了是-1。
    观察到的是一个三元组：玩家当前的总和，庄家显示的一张牌（1-10，其中1是A牌），以及玩家是否持有可用的A牌（0或1）。
    这个环境对应于Sutton和Barto（1998）的《强化学习：简介》中例5.1所描述的二十一点问题的版本。
    https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
    """
    def __init__(self,natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((spaces.Discrete(32),spaces.Discrete(11),spaces.Discrete(2)))
        self._seed()

        self.natural = natural
        self._reset()
        self.num_of_action = 2

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,action):
        assert self.action_space.contains(action)
        if action:
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1

            else:
                done = False
                reward = 0
        else:
            done =True
            while sum_hand(self.dealer) < 17:
                self.dealer.append((draw_card(self.np_random)))

            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(),reward,done,{}

    def _get_obs(self):
        return sum_hand(self.player), self.dealer[0], usable_ace(self.player)

    def _reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        # Auto-draw another card if the score is less than 12
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return self._get_obs()

