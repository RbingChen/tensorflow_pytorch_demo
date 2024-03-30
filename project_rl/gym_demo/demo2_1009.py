# coding:utf-8

"""
    参考文档：
    https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
    https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/
"""

import gym
import time


def case1():
    env = gym.make("Taxi-v3", render_mode="human")
    print("Observation Space: ", env.observation_space)
    print("Action Space       ", env.action_space)
    print("reward Table : ", env.P)
    obs = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, trun, info = env.step(action)
        env.render()
        time.sleep(0.01)
        if done:
            env.reset()
    env.close()


# case1()
def case2():
    env = gym.make("Taxi-v3", render_mode="ansi")
    obs = env.reset()
    print("Observation Space: ", env.observation_space)
    print("Action Space       ", env.action_space)
    env.s = 328
    epoch = 0
    penalties, reward = 0, 0
    frames = []
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, trun, info = env.step(action)
        if reward == -10:
            penalties += 1
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
        })
        epoch += 1
    print("Timesteps taken: {}".format(epoch))
    print("Penalties incurred: {}".format(penalties))

    from IPython.display import clear_output
    from time import sleep

    def print_frames(frames):
        for i, frame in enumerate(frames):
            clear_output(wait=True)
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.1)
    print_frames(frames)

case2()
