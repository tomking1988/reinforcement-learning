

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        greedy_action = np.argmax(Q[observation])
        probabilities = np.empty(nA)
        probabilities.fill(epsilon / nA)
        probabilities[greedy_action] = 1.0 - epsilon + epsilon / nA
        return probabilities
        # Implement this!

    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(1, num_episodes + 1):
        print("Running episode " + str(i_episode))
        observation = env.reset()
        hasMet = defaultdict(float)
        for t in range(100):
            action_probabilities = policy(observation)
            action = np.random.choice(len(action_probabilities), p=action_probabilities)
            if not (observation, action) in hasMet:
                hasMet[(observation, action)] = 0.0
            observation_new, reward, done, _ = env.step(action)
            for key in hasMet:
                hasMet[key] = hasMet[key] + reward
            observation = observation_new
            if done:
                break
        for key in hasMet:
            if not key in returns_count:
                returns_count[key] = 1.0
            else:
                returns_count[key] = returns_count[key] + 1
            if not key in returns_sum:
                returns_sum[key] = hasMet[key]
            else:
                returns_sum[key] = returns_sum[key] + hasMet[key]
        for key in returns_sum:
            state, action = key
            Q[state][action] = returns_sum[key] / returns_count[key]

    return Q, policy

    # Generate an episode.
    # An episode is

Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")