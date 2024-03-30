#coding:utf-8
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ppo import PPO
import rl_utils

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 32
gamma = 0.98
lmbda = 0.95
epochs = 2
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)