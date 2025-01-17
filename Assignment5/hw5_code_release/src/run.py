import matplotlib.pyplot as plt
import numpy as np
import gym
import envs
import os.path as osp
from datetime import datetime
from tensorboardX import SummaryWriter
import sys

import torch
from agent import Agent, RandomPolicy
from mpc import MPC
from model import PENN

# Training params
TASK_HORIZON = 40
PLAN_HORIZON = 5

# CEM params
POPSIZE = 200
NUM_ELITES = 20
MAX_ITERS = 5

# Model params
LR = 1e-3

# Dims
STATE_DIM = 8

LOG_DIR = './data'


class ExperimentGTDynamics(object):
    def __init__(self, env_name='Pushing2D-v1', mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        # Does not need model
        self.warmup = False
        mpc_params['use_gt_dynamics'] = True
        self.cem_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, use_random_optimizer=False, **mpc_params)
        self.random_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, use_random_optimizer=True, **mpc_params)

    def test(self, num_episodes, optimizer='cem'):
        samples = []
        for j in range(num_episodes):
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == 'cem' else self.random_policy
                )
            )
            print('Test episode {}: {}'.format(j, samples[-1]["rewards"][-1]))
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success


class ExperimentModelDynamics:
    def __init__(self, env_name='Pushing2D-v1', num_nets=1, mpc_params=None):
        self.env = gym.make(env_name)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')


        self.task_horizon = TASK_HORIZON

        # Tensorboard logging.
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.environment_name = "pusher"
        self.logdir = 'logs/%s/%s' % (self.environment_name, self.timestamp)
        self.summary_writer = SummaryWriter(self.logdir)

        self.agent = Agent(self.env)
        mpc_params['use_gt_dynamics'] = False
        self.model = PENN(num_nets, STATE_DIM, len(self.env.action_space.sample()), LR, self.device, self.summary_writer, self.timestamp, self.environment_name)
        self.cem_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, use_random_optimizer=False, **mpc_params)
        self.random_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, use_random_optimizer=True, **mpc_params)
        self.random_policy_no_mpc = RandomPolicy(len(self.env.action_space.sample()))

    def test(self, num_episodes, optimizer='cem'):
        samples = []
        for j in range(num_episodes):
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == 'cem' else self.random_policy
                )
            )
            print('Test episode {}: {}'.format(j, samples[-1]["rewards"][-1]))
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        print('MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_return, avg_success))
        return avg_return, avg_success

    def model_warmup(self, num_episodes, num_epochs):
        """ Train a single probabilistic model using a random policy """

        samples = []
        for i in range(num_episodes):
            print("Warmup Episode %d" % (i+1))
            samples.append(self.agent.sample(self.task_horizon, self.random_policy_no_mpc))

        self.cem_policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples],
            epochs=num_epochs
        )

    def train(self, num_train_epochs, num_episodes_per_epoch, evaluation_interval):
        """ Jointly training the model and the policy """
        samples = []
        for i in range(num_train_epochs):
            print("####################################################################")
            print("Starting training epoch %d." % (i + 1))

            
            for j in range(num_episodes_per_epoch):
                new_sample = self.agent.sample(
                        self.task_horizon, self.cem_policy
                    )
                samples.append(new_sample)
            if(len(samples)>10*num_episodes_per_epoch):
                samples = samples[num_episodes_per_epoch:]
            print("Rewards obtained:", [sample["reward_sum"] for sample in samples])

            self.cem_policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples],
                epochs=5
            )

            if (i + 1) % evaluation_interval == 0:
                cem_avg_return, cem_avg_success = self.test(20, optimizer='cem')
                print('Test success CEM + MPC:', cem_avg_success)
                rand_avg_return, rand_avg_success = self.test(20, optimizer='random')
                print('Test success Random + MPC:', rand_avg_success)

                self.summary_writer.add_scalar("test/CEM-AverageSuccess", cem_avg_success, i)
                self.summary_writer.add_scalar("test/Rand-AverageSuccess", rand_avg_success, i)
                self.summary_writer.add_scalar("test/CEM-AverageReturn", cem_avg_return, i)
                self.summary_writer.add_scalar("test/Rand-AverageReturn", rand_avg_return, i)


def test_cem_gt_dynamics(num_episode=10):
    # mpc_params = {'use_mpc': False, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('CEM PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    # mpc_params = {'use_mpc': True, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    
    # mpc_params = {'use_mpc': False, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode)
    # print('CEM PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))
    #
    mpc_params = {'use_mpc': True, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name='Pushing2DNoisyControl-v1', mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode)
    print('MPC PushingEnv Noisy: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    # mpc_params = {'use_mpc': False, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    # print('MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))

    # mpc_params = {'use_mpc': True, 'num_particles': 1}
    # exp = ExperimentGTDynamics(env_name='Pushing2D-v1', mpc_params=mpc_params)
    # avg_reward, avg_success = exp.test(num_episode, optimizer='random')
    # print('MPC PushingEnv: avg_reward: {}, avg_success: {}'.format(avg_reward, avg_success))


def train_single_dynamics(num_test_episode=50):
    num_nets = 1
    num_episodes = 1000
    num_epochs = 100
    mpc_params = {'use_mpc': True, 'num_particles': 6} #temp
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params)
    exp.model_warmup(num_episodes=num_episodes, num_epochs=num_epochs)

    avg_reward, avg_success = exp.test(num_test_episode, optimizer='cem')


def train_pets():
    num_nets = 2
    num_epochs = 500
    evaluation_interval = 50
    num_episodes_per_epoch = 1

    mpc_params = {'use_mpc': True, 'num_particles': 6}
    exp = ExperimentModelDynamics(env_name='Pushing2D-v1', num_nets=num_nets, mpc_params=mpc_params)
    exp.model_warmup(num_episodes=100, num_epochs=10)
    exp.train(num_train_epochs=num_epochs,
              num_episodes_per_epoch=num_episodes_per_epoch,
              evaluation_interval=evaluation_interval)


if __name__ == "__main__":
    # test_cem_gt_dynamics(50)
    if(len(sys.argv[1:])<1):
        print("Noob enter 0 or 1")
    elif(sys.argv[1] == '0'):
        train_single_dynamics(50)
    elif(sys.argv[1] == '1'):
        train_pets()