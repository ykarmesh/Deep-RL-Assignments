from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import deeprl_hw2q2.lake_envs as lake_env
import time
from deeprl_hw2q2.rl import policy_iteration_sync
from deeprl_hw2q2.rl import evaluate_policy_sync

def main():
    # create the environment
    env = gym.make('Deterministic-4x4-FrozenLake-v0')
    gamma = 0.9

    policy, value_func, policy_iters, value_iters =  policy_iteration_sync(env, gamma)
    # print('Agent received total reward of: %f' % total_reward)
    print('Value function took %d steps' % value_iters)


if __name__ == '__main__':
    main()