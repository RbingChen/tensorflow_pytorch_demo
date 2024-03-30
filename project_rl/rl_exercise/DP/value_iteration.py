# coding:utf-8

import numpy as np
import pprint
import sys

if "../" not in sys.path:
    sys.path.append("../")
from envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)

env = GridworldEnv()

#env.render()


def value_iteration(env, theta=0.001, discount_factor=1.0):
    """

    """

    def one_step_lookahead(state, V):
        A = np.zeros(env.num_of_action)
        for a in range(env.num_of_action):
            #print(env.transition_proba[state][a])
            for prob, next_state, reward, done in env.transition_proba[state][a]:
            #print(env.transition_proba[state][a])
                A[a] += prob * (reward + discount_factor * V[next_state])#更新的时候用的上一个状态的 值函数吗？
        #print(A)
        return A

    V = np.zeros(env.num_of_state)
    for kk in range(10):
        print(V)
        delta = 0
        for s in range(env.num_of_state):
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            #print(best_action_value)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros([env.num_of_state, env.num_of_action])
    for s in range(env.num_of_state):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0

    return policy, V

env.render()
print(env.transition_proba)
policy, v = value_iteration(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
