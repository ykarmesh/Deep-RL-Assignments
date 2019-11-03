import os
import copy
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
from tensorboardX import SummaryWriter

from algo.ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork, CriticNetworkTD3

BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA = 0.98                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001


class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the class.
        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        self.call_count = 0

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        # self.epsilon = max((1-0.9 * self.call_count/100000),0.1)
        self.call_count += 1
        if np.random.uniform() > self.epsilon:
            return np.clip(action + np.random.normal(self.mu, self.sigma, 2), -1.0, 1.0)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, args, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        np.random.seed(1337)

        self.env = env
        self.args = args
        self.outfile = outfile_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.weights_path = 'models/%s' % (self.timestamp)

        # Data for plotting.
        self.rewards_data = []  # n * [epoch, mean(returns), std(returns)]
        self.count = 0

        self.action_selector = EpsilonNormalActionNoise(0, 0.05, self.args.epsilon)
        self.memory = ReplayBuffer(args.buffer_size, args.burn_in, state_dim, action_dim, self.device)
        self.actor = ActorNetwork(state_dim, action_dim, self.args.batch_size, self.args.tau, self.args.actor_lr, self.device, args.custom_init)
        if self.args.algorithm == 'ddpg' or self.args.algorithm == 'her':
            self.critic = CriticNetwork(state_dim, action_dim, self.args.batch_size,  \
                self.args.tau, self.args.critic_lr, self.args.gamma, self.device, args.custom_init)
        elif self.args.algorithm == 'td3':
            self.critic = CriticNetworkTD3(state_dim, action_dim, self.args.batch_size,  \
                self.args.tau, self.args.critic_lr, self.args.gamma, self.device, args.custom_init)

        if args.weights_path: self.load_model()

        if args.train:
            # Tensorboard logging.
            self.logdir = 'logs/%s' % (self.timestamp)
            self.imgdir = 'imgs/%s' % (self.timestamp)
            os.makedirs(self.imgdir)
            self.summary_writer = SummaryWriter(self.logdir)

            # Save hyperparameters.
            with open(self.logdir + '/training_parameters.json', 'w') as f:
                json.dump(vars(self.args), f, indent=4)

    def save_model(self, epoch):
        '''Helper function to save model state and weights.'''
        if not os.path.exists(self.weights_path): os.makedirs(self.weights_path)
        torch.save({'policy_state_dict': self.actor.policy.state_dict(),
                    'policy_target_state_dict': self.actor.policy_target.state_dict(),
                    'policy_optimizer': self.actor.policy_optimizer.state_dict(),
                    'critic_state_dict': self.critic.critic.state_dict(),
                    'critic_target_state_dict': self.critic.critic_target.state_dict(),
                    'critic_optimizer': self.critic.critic_optimizer.state_dict(),
                    'rewards_data': self.rewards_data,
                    'epoch': epoch},
                    os.path.join(self.weights_path, 'model_%d.h5' % epoch))

    def load_model(self):
        '''Helper function to load model state and weights. '''
        if os.path.isfile(self.args.weights_path):
            print('=> Loading checkpoint', self.args.weights_path)
            self.checkpoint = torch.load(self.args.weights_path)
            self.actor.policy.load_state_dict(self.checkpoint['policy_state_dict'])
            self.actor.policy_target.load_state_dict(self.checkpoint['policy_target_state_dict'])
            self.actor.policy_optimizer.load_state_dict(self.checkpoint['policy_optimizer'])
            self.critic.critic.load_state_dict(self.checkpoint['critic_state_dict'])
            self.critic.critic_target.load_state_dict(self.checkpoint['critic_target_state_dict'])
            self.critic.critic_optimizer.load_state_dict(self.checkpoint['critic_optimizer'])
            self.rewards_data = self.checkpoint['rewards_data']
        else:
            raise Exception('No checkpoint found at %s' % self.args.weights_path)

    def evaluate(self, num_episodes):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                with torch.no_grad():
                    a_t = self.actor.policy(torch.tensor(s_t, device=self.device).float())
                new_s, r_t, done, info = self.env.step(a_t.cpu().numpy())
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            if i < 9 and self.args.render:
                plt.subplot(3, 3, i+1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    plt.savefig(os.path.join(self.imgdir,str(self.count)))
                    self.count += 1
                    # plt.show()
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

    def train(self, num_episodes):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
        """
        for i in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            step = 0
            critic_loss = 0
            trajectory_data = []
            state = torch.tensor(state, device=self.device).float()

            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                with torch.no_grad():
                    action = self.actor.policy(state)
                    env_action = self.action_selector(action.cpu().numpy())
                    action = torch.tensor(env_action, device=self.device).float()

                next_state, reward, done, info = self.env.step(env_action)
                next_state = torch.tensor(next_state, device=self.device).float()

                self.memory.add(state, action, torch.tensor(reward, device=self.device),
                    next_state, torch.tensor(done, device=self.device))

                # Save data for HER.
                if self.args.algorithm == 'her':
                    trajectory_data.append([state.detach().cpu().numpy(), action.detach().cpu().numpy(),
                        reward, next_state.detach().cpu().numpy(), done])

                total_reward += reward
                step += 1

                if not done:
                    state = next_state.clone().detach()

            if self.args.algorithm == 'her':
                # For HER, we also want to save the final next_state.
                self.add_hindsight_replay_experience(trajectory_data)

            if self.memory.burned_in:
                if self.args.algorithm in ['ddpg', 'her']:
                    critic_loss, policy_loss, new_metric = self.train_DDPG()
                    self.summary_writer.add_scalar('train/policy_loss', policy_loss, i)
                    self.summary_writer.add_scalar('train/new_metric', new_metric.mean(), i)

                elif self.args.algorithm == 'td3':
                    critic_loss, policy_loss = self.train_TD3(i)
                    if i % self.args.policy_update_frequency == 0:
                        self.summary_writer.add_scalar('train/policy_loss', policy_loss, i)

            # Logging
            if self.memory.burned_in and i % self.args.log_interval == 0:
                print("Episode %d: Total reward = %d" % (i, total_reward))
                print("\tTD loss = %.2f" % (critic_loss / step,)) 
                # print("\tSteps = %d; Info = %s" % (step, info['done']))

                self.summary_writer.add_scalar('train/trajectory_length', step, i)
                self.summary_writer.add_scalar('train/critic_loss', critic_loss, i)

            if i % self.args.test_interval == 0:
                successes, mean_rewards, std_rewards = self.evaluate(10)
                self.rewards_data.append([i, mean_rewards, std_rewards])

                self.summary_writer.add_scalar('test/success', successes, i)
                self.summary_writer.add_scalar('test/rewards_mean', mean_rewards, i)
                self.summary_writer.add_scalar('test/rewards_std', std_rewards, i)

                print('Evaluation: success = %.2f; return = %.2f' % (successes, mean_rewards))
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f,\n" % (successes, mean_rewards))

            if i % self.args.save_interval == 0:
                self.save_model(i)

        self.save_model(i)
        self.summary_writer.close()

    def add_hindsight_replay_experience(self, trajectory):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        # Get new goal location (last location of box).
        goal = trajectory[-1][3][2:4]

        # Relabels a trajectory using a new goal state.
        for state, action, reward, next_state, done in trajectory:
            state[-2:] = goal.copy()
            next_state[-2:] = goal.copy()            
            reward = self.env._HER_calc_reward(state)
            if reward == 0: done = True

            self.memory.add(
                torch.tensor(state, device=self.device),
                torch.tensor(action, device=self.device),
                torch.tensor(reward, device=self.device),
                torch.tensor(next_state, device=self.device),
                torch.tensor(done, device=self.device))

            if reward == 0: break

    def train_DDPG(self):
        for j in range(self.args.num_update_iters):
            states, actions, rewards, next_states, dones = self.memory.get_batch(self.args.batch_size)
            next_actions = self.actor.policy_target(next_states).detach()
            critic_loss = self.critic.train(states, actions, rewards, next_states, dones, next_actions)
            new_Q_value = self.critic.critic(states, self.actor.policy(states))
            policy_loss = self.actor.train(new_Q_value)

            self.critic.update_target()
            self.actor.update_target()

        return critic_loss, policy_loss, new_Q_value
    
    def train_TD3(self, i):
        for j in range(self.args.num_update_iters):
            states, actions, rewards, next_states, dones = self.memory.get_batch(self.args.batch_size)
            next_actions = self.noise_regularization(self.actor.policy_target(next_states).detach().cpu().numpy())
            next_actions = torch.tensor(next_actions, device=self.device).float()

            critic_loss = self.critic.train(states, actions, rewards, next_states, dones, next_actions)

            policy_loss = 0
            if (i*self.args.num_update_iters + j)%self.args.policy_update_frequency == 0:
                policy_loss = self.actor.train(self.critic.critic.get_Q(states, self.actor.policy(states)))

                self.critic.update_target()
                self.actor.update_target()

        return critic_loss, policy_loss

    def noise_regularization(self, next_actions):
        return np.clip(next_actions + np.clip(np.random.normal(0, self.args.target_action_sigma, (self.args.batch_size, 2)), -self.args.clip, self.args.clip), -1.0, 1.0)
        # return next_actions

    def plot(self):
        # Save the plot.
        filename = os.path.join('plots', *self.args.weights_path.split('/')[-2:]).replace('.h5', '.png')
        if not os.path.exists(os.path.dirname(filename)): os.makedirs(os.path.dirname(filename))

        # Make error plot with mean, std of rewards.
        data = np.asarray(self.rewards_data)
        plt.errorbar(data[:, 0], data[:, 1], data[:, 2], lw=2.5, elinewidth=1.5,
            ecolor='grey', barsabove=True, capthick=2, capsize=3)
        plt.title('Cumulative Rewards (Mean/Std) Plot for A2C Algorithm')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Cumulative Rewards')
        plt.grid()
        plt.savefig(filename, dpi=300)
        plt.show()
