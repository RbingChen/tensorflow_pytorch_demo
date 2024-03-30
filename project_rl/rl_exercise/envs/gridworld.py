# coding:utf-8

import io
import numpy as np
import sys
from . import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    """
    这是来自Sutton的强化学习书籍第4章的网格世界环境。
    你是一个在MxN网格上的代理，你的目标是到达左上角或右下角的终止状态。

    例如，一个4x4的网格如下所示：

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x是你的位置，T是两个终止状态。

    你可以在每个方向上采取行动（UP=0，RIGHT=1，DOWN=2，LEFT=3）。
    超出边缘的行动会让你停留在当前状态。
    在你到达终止状态之前，每一步你都会收到-1的奖励。
    """

    def __init__(self, shape=[4, 5]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape必须是长度为2的list 或 tunple')
        self.shape = shape

        num_of_state = np.prod(shape)
        num_of_action = 4
        MAX_Y = shape[0]
        MAX_X = shape[1]

        proba = {}
        grid = np.arange(num_of_state).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            state = it.iterindex
            y, x = it.multi_index
            proba[state] = {a: [] for a in range(num_of_action)}

            is_done = lambda s: s == 0 or s == (num_of_state - 1)
            reward = 0.0 if is_done(state) else -1.0

            if is_done(state):
                proba[state][UP] = [(1.0, state, reward, True)]
                proba[state][RIGHT] = [(1.0, state, reward, True)]
                proba[state][DOWN] = [(1.0, state, reward, True)]
                proba[state][LEFT] = [(1.0, state, reward, True)]
            # 没有terminal 状态
            else:
                ns_up = state if y == 0 else state - MAX_X
                ns_right = state if x == (MAX_X - 1) else state + 1
                ns_down = state if y == (MAX_Y - 1) else state + MAX_X
                ns_left = state if x == 0 else state - 1
                proba[state][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                proba[state][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                proba[state][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                proba[state][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        init_state_distribution = np.ones(num_of_state) / num_of_state
        super(GridworldEnv, self).__init__(num_of_state, num_of_action, proba, init_state_distribution)

    def render(self, mode="human", close=False):
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.num_of_state).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            state = it.iterindex
            y, x = it.multi_index

            if self.state == state:
                output = " x "
            elif state == 0 or state == self.num_of_state - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
