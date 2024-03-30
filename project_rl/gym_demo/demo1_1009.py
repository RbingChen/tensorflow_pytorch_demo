# coding:utf-8

import gym
import matplotlib.pyplot as plt

"""
  见博客：
  https://blog.paperspace.com/getting-started-with-openai-gym/
"""


def case1():
    env = gym.make("MountainCar-v0", new_step_api=True)
    # obs_space = env.observation_space
    # action_space = env.action_space
    # print("The observation space: {}".format(obs_space))
    # print("The action space: {}".format(action_space))

    obs = env.reset()
    print("The initial observation is {} {}".format(obs, type(obs)))
    random_action = env.action_space.sample()
    print("random_action:{}".format(random_action))

    new_obs, reward, done, trun, info = env.step(random_action)
    print("The new observation is {}".format(new_obs))
    # env.render(mode="human")
    env_screen = env.render(mode='rgb_array')
    env.close()
    plt.imshow(env_screen)


"""
 爬山汽车的动作是什么？
"""
import time


def case2():
    env = gym.make("MountainCar-v0")
    obs = env.reset()
    print(type(obs)) #<class 'tuple'>
    num_steps = 2
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, trun, info = env.step(action)
        print(obs)
        print(type(obs)) #<class 'numpy.ndarray'>
        env.render()
        time.sleep(0.001)
        if done:
            env.reset()
    env.close()


case2()

def case3():
    env = gym.make("BreakoutNoFrameskip-v4",render_mode='human')
    print("Observation Space: ", env.observation_space)
    print("Action Space       ", env.action_space)
    obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, trun, info = env.step(action)
        env.render()
        time.sleep(0.01)
        if done:
            env.reset()
    env.close()


#case3()

def case4():
    from collections import deque
    from gym import spaces
    import numpy as np

    class ConcatObs(gym.Wrapper):
        def __init__(self, env, k):
            gym.Wrapper.__init__(self, env)
            self.k = k
            self.frames = deque([], maxlen = k)
            shp = env.observation_space.shape
            self.observation_space = spaces.Box(low=0, high=255, shape=((k,1)+shp),dtype=env.observation_space.dtype)

        def reset(self):
            ob = self.env.reset()
            for _ in range(self.k):
                self.frames.append(ob)
            return self._get_ob()

        def step(self,action):
            obs, reward, done, trun, info = self.env.step(action)
            self.frames.append(obs)
            return self._get_ob(),reward,done,info

        def _get_ob(self):
            return np.array(self.frames)


