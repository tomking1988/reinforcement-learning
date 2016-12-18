import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    while True:
        delta = 0.0
        for s in range(env.nS):
            v_new = -100
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v_new = max(v_new, V[next_state] + discount_factor * reward)
            delta = max(delta, np.abs(v_new - V[s]))
            V[s] = v_new
        print(V)
        print('delta ' + str(delta))
        if delta < theta:
            break
    # Implement!
    return policy, V

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