from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import deeprl_hw2q2.lake_envs as lake_env
import time
from deeprl_hw2q2.rl import policy_iteration_sync, value_iteration_sync, policy_iteration_async_ordered, \
    policy_iteration_async_randperm, value_iteration_async_ordered, value_iteration_async_randperm, env_wrapper

def main():
    # create the environment
    env = env_wrapper('Deterministic-8x8-FrozenLake-v0')
    gamma = 0.9
    method = 'policy_iteration'
    policy_iters = None

    if method == 'policy_iteration':
        policy, value_func, policy_iters, value_iters =  policy_iteration_sync(env, gamma)
    elif method == 'value_iteration':
        value_func, value_iters = value_iteration_sync(env, gamma)
    elif method == 'policy_iteration_async_ordered':
        policy, value_func, policy_iters, value_iters = policy_iteration_async_ordered(env, gamma)
    elif method == 'policy_iteration_async_randperm':
        policy, value_func, policy_iters, value_iters = policy_iteration_async_randperm(env, gamma)
    elif method == 'value_iteration_async_ordered':
        value_func, value_iters = value_iteration_async_ordered(env, gamma)
    elif method == 'value_iteration_async_randperm':
        value_func, value_iters = value_iteration_async_randperm(env, gamma)
    print('Value function took %d steps' % value_iters)
    if policy_iters is not None:
        print('Policy function took %d steps' % policy_iters)

if __name__ == '__main__':
    main()