'''
Main script to test LQR and iLQR on TwoLinkArmEnv-v0
'''

import os
import sys
import time
from copy import deepcopy

import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import deeprl_hw6
from controllers import calc_lqr_input
from ilqr import calc_ilqr_input


class Agent:
    def __init__(self, env_name='TwoLinkArm-v0', policy='LQR'):
        self.env = gym.make(env_name)
        self.sim_env = deepcopy(self.env) #gym.make(env_name)
        self.env_name = env_name
        self.algo = policy

        # Folder for plots
        self.folder = os.path.join(env_name, policy)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run_LQR(self):
        rewards = []
        states, actions = [self.env.reset()], []
        self.sim_env.reset()

        count = 1
        while True:
            action = calc_lqr_input(self.env, self.sim_env)
            state, reward, done, info = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # self.env.render()
            # time.sleep(0.2)

            # Display step
            sys.stdout.write('\rSteps: %04d | Reward: %d\t' % (count, reward))
            sys.stdout.flush()
            if done: break
            count += 1

        print('\nRewards Sum:', np.sum(rewards))
        print('Rewards Mean:', np.mean(rewards))

        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
        }

        # self.plot(trajectory)
        return np.array(actions).T, states

    def run_iLQR(self):
        tN = 100
        rewards = []
        U, LQR_X = self.run_LQR()
        U = U[:, :tN]
        self.env.state = LQR_X[0]
        states, actions = [LQR_X[0]], []
        
        U = np.zeros((self.env.action_space.shape[0], tN))*100
        
        self.sim_env = deepcopy(self.env)

        count = 0
        i = 0

        U, costs = calc_ilqr_input(self.env, self.sim_env, U, tN=tN)
        while count < tN:
            # if i >= tN:
            #     print('\nRewards Sum:', np.sum(rewards))
            #     U = calc_ilqr_input(self.env, self.sim_env, U, tN=tN)
            #     i = 0 
            # U = calc_ilqr_input(self.env, self.sim_env, U, tN=tN)
            state, reward, done, info = self.env.step(U[:, count])

            time.sleep(0.1)
            self.env.render()

            states.append(state)
            rewards.append(reward)
            actions.append(U[:, count])

            # Display step
            sys.stdout.write('\rSteps: %04d | Reward: %d\t' % (count, reward))
            sys.stdout.flush()
            if done: break
            count += 1
            i += 1

        print('\nRewards Sum:', np.sum(rewards))
        print('Rewards Mean:', np.mean(rewards))

        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
        }

        self.plot(trajectory, costs)

    def run(self):
        if self.algo == 'LQR':
            self.run_LQR()
        elif self.algo == 'iLQR':
            print("Starting iLQR")
            self.run_iLQR()
        else:
            print("Wrong Algorithm selected: {}".format(self.policy))


    def plot(self, trajectory, costs=None):
        total = len(trajectory['rewards'])

        # Joint angles plot
        plt.title(r'%s: Joint Angles (q)' % self.env_name)
        import pdb; pdb.set_trace()
        plt.plot(trajectory['states'][:, 0], label=r'$q_1$')
        plt.plot(trajectory['states'][:, 1], label=r'$q_2$')
        plt.xlabel('Steps (Total: %d)' % total)
        plt.ylabel('Joint Angles (rad)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.folder, 'joint_angles.png'), dpi=300)

        # Joint velocities plot
        plt.figure()
        plt.title(r'%s: Joint Velocities $(\dot{q})$' % self.env_name)
        plt.plot(trajectory['states'][:, 2], label=r'$\dot{q}_1$')
        plt.plot(trajectory['states'][:, 3], label=r'$\dot{q}_2$')
        plt.xlabel('Steps (Total: %d)' % total)
        plt.ylabel('Joint Velocities (rad/s)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.folder, 'joint_velocities.png'), dpi=300)

        # Control inputs plot
        plt.figure()
        plt.title(r'%s: Control Inputs (u)' % self.env_name)
        plt.plot(trajectory['actions'][:, 0], label=r'$u_1$')
        plt.plot(trajectory['actions'][:, 1], label=r'$u_2$')
        plt.xlabel('Steps (Total: %d)' % total)
        plt.ylabel('Control Inputs')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.folder, 'control_inputs.png'), dpi=300)

        # cost
        if costs is not None:
            plt.figure()
            plt.title(r'%s: Cost (u)' % self.env_name)
            plt.plot(costs[:], label=r'$u_1$')
            plt.xlabel('Steps (Total: %d)' % total)
            plt.ylabel('Cost')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.folder, 'cost.png'), dpi=300)


if __name__ == '__main__':
    agent = Agent(policy='iLQR')
    agent.run()
