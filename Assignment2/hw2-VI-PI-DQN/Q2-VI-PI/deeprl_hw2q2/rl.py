# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import deeprl_hw2q2.lake_envs as lake_env
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import gym


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_func):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    # Hint: You might want to first calculate Q value,
    #       and then take the argmax.
    policy = np.zeros(env.nS, dtype='int')
    for state in range(env.nS):
      max_value = -np.inf
      best_action = -1
      for action in range(env.nA):
        value = 0
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
          prob = env.T[state, action, nextstate]
          reward = env.R[state, action, nextstate]
          value += prob*(reward + gamma*(1-int(is_terminal))*value_func[nextstate])
        if max_value < value:
          max_value = value
          best_action = action

        policy[state] = best_action
    return policy

# Question 2.1
def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    next_value_func = np.zeros(env.nS)
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
      delta = np.zeros(env.nS)
      for state in range(env.nS):
        # Find out the current action encoded in the policy
        action = policy[state]
        # Iterate over all the future states
        new_value = 0
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
          prob = env.T[state, action, nextstate]
          reward = env.R[state, action, nextstate]
          new_value += prob*(reward + gamma*(1-int(is_terminal))*value_func[nextstate])
        
        delta[state] = max(delta[state], abs(next_value_func[state] - new_value))
        next_value_func[state] = new_value
      value_func = next_value_func.copy()
      iters += 1
    return value_func, iters


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
      delta = np.zeros(env.nS)
      for state in range(env.nS):
        # Find out the current action encoded in the policy
        action = policy[state]
        # Iterate over all the future states
        new_value = 0
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
          prob = env.T[state, action, nextstate]
          reward = env.R[state, action, nextstate]
          new_value += prob*(reward + gamma*(1-int(is_terminal))*value_func[nextstate])
        
        delta[state] = max(delta[state], abs(value_func[state] - new_value))
        value_func[state] = new_value
      iters += 1
    return value_func, iters


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
      delta = np.zeros(env.nS)
      states = np.random.choice(env.nS, env.nS, replace=False)

      for state in states:
        # Find out the current action encoded in the policy
        action = policy[state]
        # Iterate over all the future states
        new_value = 0
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
          prob = env.T[state, action, nextstate]
          reward = env.R[state, action, nextstate]
          new_value += prob*(reward + gamma*(1-int(is_terminal))*value_func[nextstate])
        
        delta[state] = max(delta[state], abs(value_func[state] - new_value))
        value_func[state] = new_value
      iters += 1
    return value_func, iters

# Question 2.1
def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
    for state in range(env.nS):
      max_value = -np.inf
      best_action = -1
      for action in range(env.nA):
        value = 0
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
          prob = env.T[state, action, nextstate]
          reward = env.R[state, action, nextstate]
          value += prob*(reward + gamma*(1-int(is_terminal))*value_func[nextstate])
        if max_value < value:
          max_value = value
          best_action = action

      if policy[state] != best_action:
        policy_stable = False
        policy[state] = best_action
    print("Policy {}".format(policy))
    return policy_stable, policy

# Question 2.1
def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    display_policy_letters(env, policy)
    policy_stable = False
    value_iters = 0
    policy_iters = 0
    iters = 0
    while not policy_stable:
      iters += 1
      value_func, i = evaluate_policy_sync(env, gamma, policy)
      value_iters += i
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      policy_iters += 1
      # pdb.set_trace()
      print("iters {} | policy eval {} | policy improvement {}".format(iters, value_iters, policy_iters))
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    print("Policy Evaluation Complete \n{}".format(value_func))
    return policy, value_func, policy_iters, value_iters


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    display_policy_letters(env, policy)
    policy_stable = False
    value_iters = 0
    policy_iters = 0
    iters = 0
    while not policy_stable:
      iters += 1
      value_func, i = evaluate_policy_async_ordered(env, gamma, policy)
      value_iters += i
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      policy_iters += 1
      # pdb.set_trace()
      print("iters {} | policy eval {} | policy improvement {}".format(iters, value_iters, policy_iters))
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    print("Policy Evaluation Complete \n{}".format(value_func))
    return policy, value_func, policy_iters, value_iters


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    display_policy_letters(env, policy)
    policy_stable = False
    value_iters = 0
    policy_iters = 0
    iters = 0
    while not policy_stable:
      iters += 1
      value_func, i = evaluate_policy_async_randperm(env, gamma, policy)
      value_iters += i
      policy_stable, policy = improve_policy(env, gamma, value_func, policy)
      policy_iters += 1
      # pdb.set_trace()
      print("iters {} | policy eval {} | policy improvement {}".format(iters, value_iters, policy_iters))
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    print("Policy Evaluation Complete \n{}".format(value_func))
    return policy, value_func, policy_iters, value_iters

# Question 2.4
def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    next_value_func = np.zeros(env.nS)
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
      delta = np.zeros(env.nS)
      for state in range(env.nS):
        max_value = -np.inf
        for action in range(env.nA):
          # Iterate over all the future states
          new_value = 0
          for prob, nextstate, reward, is_terminal in env.P[state][action]:
            prob = env.T[state, action, nextstate]
            reward = env.R[state, action, nextstate]
            new_value += prob*(reward + gamma*(1-int(is_terminal))*value_func[nextstate])

          if max_value < new_value:
            max_value = new_value

        delta[state] = max(delta[state], abs(value_func[state] - max_value))
        next_value_func[state] = max_value
      value_func = next_value_func.copy()
      iters += 1
    print("Policy Evaluation Complete \n{}".format(value_func))

    policy = value_function_to_policy(env, gamma, value_func)
    
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    return value_func, iters


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
      delta = np.zeros(env.nS)
      for state in range(env.nS):
        max_value = -np.inf
        for action in range(env.nA):
          # Iterate over all the future states
          new_value = 0
          for prob, nextstate, reward, is_terminal in env.P[state][action]:
            prob = env.T[state, action, nextstate]
            reward = env.R[state, action, nextstate]
            new_value += prob*(reward + gamma*(1-int(is_terminal))*value_func[nextstate])

          if max_value < new_value:
            max_value = new_value

        delta[state] = max(delta[state], abs(value_func[state] - max_value))
        value_func[state] = max_value
      iters += 1
    print("Policy Evaluation Complete \n{}".format(value_func))

    policy = value_function_to_policy(env, gamma, value_func)
    
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    return value_func, iters



def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    iters = 0
    delta = np.ones(env.nS)
    while iters < max_iterations and np.any((delta > tol)):
      delta = np.zeros(env.nS)
      states = np.random.choice(env.nS, env.nS, replace=False)

      for state in states:
        max_value = -np.inf
        for action in range(env.nA):
          # Iterate over all the future states
          new_value = 0
          for prob, nextstate, reward, is_terminal in env.P[state][action]:
            prob = env.T[state, action, nextstate]
            reward = env.R[state, action, nextstate]
            new_value += prob*(reward + gamma*(1-int(is_terminal))*value_func[nextstate])

          if max_value < new_value:
            max_value = new_value

        delta[state] = max(delta[state], abs(value_func[state] - max_value))
        value_func[state] = max_value
      iters += 1
    print("Policy Evaluation Complete \n{}".format(value_func))

    policy = value_function_to_policy(env, gamma, value_func)
    
    display_policy_letters(env, policy)
    value_func_heatmap(env, value_func)
    return value_func, iters



def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    return value_func, 0


######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy):
    """Displays a policy as letters, as required by problem 2.2 & 2.6

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])
    
    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)
    

    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)
    
    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))
    
    for state in range(env.nS):
      for action in range(env.nA):
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
            env.T[state, action, nextstate] = prob
            env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func):
    """Visualize a policy as a heatmap, as required by problem 2.3 & 2.5

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7,6)) 
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]), 
                annot=False, linewidths=.5, cmap="YlGnBu", ax=ax,
                yticklabels = np.arange(1, env.nrow+1)[::-1], 
                xticklabels = np.arange(1, env.nrow+1)) 
    plt.show()
    # Other choices of cmap: YlGnBu
    # More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    return None

    