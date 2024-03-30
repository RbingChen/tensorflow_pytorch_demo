# coding:utf-8

import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("Breakout-v0",render_mode="rgb_array")
print("Action space size: {}".format(env.action_space.n))
print(env.get_action_meanings()) # env.unwrapped.get_action_meanings() for gym 0.8.0 or later

observation = env.reset()
print("Observation space shape: {}".format(observation[0].shape))

plt.figure()
plt.imshow(env.render())
plt.show()

[env.step(2) for x in range(1)]
plt.figure()
plt.imshow(env.render())
plt.show()
env.close()