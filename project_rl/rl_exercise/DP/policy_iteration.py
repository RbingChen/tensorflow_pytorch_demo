# coding:utf-8


import numpy as np
import pprint
import sys

if "../" not in sys.path:
    sys.path.append("../")
from envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)

env = GridworldEnv()


def policy_evalution(policy, env, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.num_of_state)
    while True:
        delta = 0
        for state in range(env.num_of_state):
            v = 0
            for a, action_proba in enumerate(policy[state]):

                for prob, next_state, reward, done in env.transition_proba[state][a]:
                    v += action_proba * prob * (reward + discount_factor * V[next_state])

            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        if delta < theta:
            break
    print("dododo")
    return V


def policy_improvement(env, policy_eval_fn=policy_evalution, discount_factor=1.0):
    def one_step_lookahead(state, V):
        A = np.zeros(env.num_of_action)
        for a in range(env.num_of_action):
            # print(env.transition_proba[state][a])
            for prob, next_state, reward, done in env.transition_proba[state][a]:
                # print(env.transition_proba[state][a])
                A[a] += prob * (reward + discount_factor * V[next_state])  # 更新的时候用的上一个状态的 值函数吗？
        # print(A)
        return A

    policy = np.ones([env.num_of_state, env.num_of_action]) / env.num_of_action

    while True:
        V = policy_eval_fn(policy,env,discount_factor)
        policy_stable = True
        for s in range(env.num_of_state):
            chosen_a = np.argmax(policy[s])
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            if chosen_a != best_a:
                policy_stable = False

            policy[s] = np.eye(env.num_of_action)[best_a]

        if policy_stable:
            return policy, V


policy, v = policy_improvement(env)
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

