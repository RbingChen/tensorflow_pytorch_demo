# coding:utf-8
import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
    sys.path.append("../")

from envs import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

env = gym.make("MountainCar-v0", render_mode="rgb_array")

print(env.observation_space.sample())
print(type(env.observation_space.sample()))

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


class Estimator():
    def __init__(self,featurizer):
        self.featurizer = featurizer
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            feat = self.featurize_state(env.reset())
            print(feat)
            model.partial_fit([feat], [0])
            self.models.append(model)

    def featurize_state(self,state):
        if isinstance(state,tuple):
            state = state[0]
        scaled = scaler.transform([state])
        featurizer = self.featurizer.transform(scaled)

        return featurizer[0]

    def predict(self,s, a=None):
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        features = self.featurize_state(s)
        self.models[a].partial_fit([features],[y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):


    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for eps in range(num_episodes):
        policy = make_epsilon_greedy_policy(estimator, epsilon*epsilon_decay**eps, env.action_space.n)

        last_reward = stats.episode_rewards[eps-1]
        sys.stdout.flush()

        state = env.reset()
        next_action = None

        for t in itertools.count():
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action

            next_state, reward, done,trun, info = env.step(action)
            stats.episode_rewards[eps] += reward
            stats.episode_lengths[eps] = t

            q_values_next = estimator.predict(next_state)

            td_target = reward + discount_factor * np.max(q_values_next)
            estimator.update(state, action, td_target)

            print("\rStep {} @ Episode {}/{} ({})".format(t, eps + 1, num_episodes, last_reward), end="")

            if done:
                break
            state = next_state
    return stats


estimator = Estimator(featurizer)
stats = q_learning(env, estimator, 100, epsilon=0.0)
plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)