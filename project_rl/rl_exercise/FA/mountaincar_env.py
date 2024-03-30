# coding:utf-8

import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("MountainCar-v0", render_mode="rgb_array")

env.reset()
plt.figure()
plt.imshow(env.render())

[env.step(0) for x in range(10000)]
plt.figure()
plt.imshow(env.render())
plt.show()

env.close()
