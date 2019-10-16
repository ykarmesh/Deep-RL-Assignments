import os
import sys
import pdb
import json
import argparse
from datetime import datetime
from collections import deque

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from env import preprocess
from model import ActorCritic


class A3C():
    '''Implementation of N-step Asychronous Advantage Actor Critic'''
    def __init__(self, args, env, train=True):
        self.args = args
        self.set_random_seeds()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the environment.
        self.env = gym.make(env)
        self.environment_name = env

        # Setup model.
        self.policy = ActorCritic(4, self.env.action_space.n)
        self.policy.apply(self.initialize_weights)

        # Setup critic model.
        self.critic = ActorCritic(4, self.env.action_space.n)
        self.critic.apply(self.initialize_weights)

        # Setup optimizer.
        self.eps = 1e-10  # To avoid divide-by-zero error.
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # Model weights path.
        self.timestamp = datetime.now().strftime('a2c-breakout-%Y-%m-%d_%H-%M-%S')
        self.weights_path = 'models/%s/%s' % (self.environment_name, self.timestamp)

        # Load pretrained weights.
        if args.weights_path: self.load_model()
        self.policy.to(self.device)
        self.critic.to(self.device)

        # Video render mode.
        if args.render:
            self.policy.eval()
            self.generate_episode(render=True)
            self.plot()
            return

        # Data for plotting.
        self.rewards_data = []  # n * [epoch, mean(returns), std(returns)]

        # Network training mode.
        if train:
            # Tensorboard logging.
            self.logdir = 'logs/%s/%s' % (self.environment_name, self.timestamp)
            self.summary_writer = SummaryWriter(self.logdir)

            # Save hyperparameters.
            with open(self.logdir + '/training_parameters.json', 'w') as f:
                json.dump(vars(self.args), f, indent=4)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def set_random_seeds(self):
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        torch.backends.cudnn.benchmark = True

    def save_model(self, epoch):
        '''Helper function to save model state and weights.'''
        if not os.path.exists(self.weights_path): os.makedirs(self.weights_path)
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    'rewards_data': self.rewards_data,
                    'epoch': epoch},
                    os.path.join(self.weights_path, 'model_%d.h5' % epoch))

    def load_model(self):
        '''Helper function to load model state and weights. '''
        if os.path.isfile(self.args.weights_path):
            print('=> Loading checkpoint', self.args.weights_path)
            self.checkpoint = torch.load(self.args.weights_path)
            self.policy.load_state_dict(self.checkpoint['policy_state_dict'])
            self.policy_optimizer.load_state_dict(self.checkpoint['policy_optimizer'])
            self.critic.load_state_dict(self.checkpoint['critic_state_dict'])
            self.critic_optimizer.load_state_dict(self.checkpoint['critic_optimizer'])
            self.rewards_data = self.checkpoint['rewards_data']
        else:
            raise Exception('No checkpoint found at %s' % self.args.weights_path)

    def train(self):
        '''Trains the model on a single episode using REINFORCE.'''
        for epoch in range(self.args.num_episodes):
            # Generate epsiode data.
            returns, log_probs, value_function, train_rewards = self.generate_episode()
            self.summary_writer.add_scalar('train/cumulative_rewards', train_rewards, epoch)
            self.summary_writer.add_scalar('train/trajectory_length', returns.size()[0], epoch)

            # Compute loss and policy gradient.
            self.policy_optimizer.zero_grad()
            policy_loss = ((returns - value_function.detach()) * -log_probs).mean()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss =  F.mse_loss(returns, value_function)
            critic_loss.backward()
            self.critic_optimizer.step()

            # Test the model.
            if epoch % self.args.test_interval == 0:
                self.policy.eval()
                print('\nTesting')
                rewards = [self.generate_episode(test=True) for epoch in range(self.args.test_episodes)]
                rewards_mean, rewards_std = np.mean(rewards), np.std(rewards)
                print('Test Rewards (Mean): %.3f | Test Rewards (Std): %.3f\n' % (rewards_mean, rewards_std))
                self.rewards_data.append([epoch, rewards_mean, rewards_std])
                self.summary_writer.add_scalar('test/rewards_mean', rewards_mean, epoch)
                self.summary_writer.add_scalar('test/rewards_std', rewards_std, epoch)
                self.policy.train()

            # Logging.
            if epoch % self.args.log_interval == 0:
                print('Epoch: {0:05d}/{1:05d} | Policy Loss: {2:.3f} | Value Loss: {3:.3f}'.format(epoch, self.args.num_episodes, policy_loss, critic_loss))
                self.summary_writer.add_scalar('train/policy_loss', policy_loss, epoch)
                self.summary_writer.add_scalar('train/critic_loss', critic_loss, epoch)

            # Save the model.
            if epoch % self.args.save_interval == 0:
                self.save_model(epoch)

        self.save_model(epoch)
        self.summary_writer.close()

    def generate_episode(self, gamma=0.99, test=False, render=False, max_iters=10000):
        '''
        Generates an episode by executing the current policy in the given env.
        Returns:
        - a list of states, indexed by time epoch
        - a list of actions, indexed by time epoch
        - a list of cumulative discounted returns, indexed by time epoch
        '''
        iters = 0
        done = False
        state = self.env.reset()

        # Set video save path if render enabled.
        if render:
            save_path = 'videos/%s/epoch-%s' % (self.environment_name, self.checkpoint['epoch'])
            if not os.path.exists(save_path): os.makedirs(save_path)
            monitor = gym.wrappers.Monitor(self.env, save_path, force=True)

        batches = []
        states = [torch.zeros(84, 84, device=self.device).float()] * 3
        rewards, returns = [], []
        actions, log_probs = [], []

        while not done:
            # Run policy on current state to log probabilities of actions.
            states.append(torch.tensor(preprocess(state), device=self.device).float().squeeze(0))
            batches.append(torch.stack(states[-4:]))
            action_probs = self.policy.forward(batches[-1].unsqueeze(0)).squeeze(0)

            # Sample action from the log probabilities.
            if test and self.args.det_eval: action = torch.argmax(action_probs)
            else: action = torch.argmax(torch.distributions.Multinomial(logits=action_probs).sample())
            actions.append(action)
            log_probs.append(action_probs[action])

            # Run simulation with current action to get new state and reward.
            if render: monitor.render()
            state, reward, done, _ = self.env.step(action.cpu().numpy())
            rewards.append(reward)

            # Break if the episode takes too long.
            iters += 1
            if iters > max_iters: break

        # Save video and close rendering.
        cum_rewards = np.sum(rewards)
        if render:
            monitor.close()
            print('\nCumulative Rewards:', cum_rewards)
            return

        # Return cumulative rewards for test mode.
        if test: return cum_rewards

        # Flip rewards from T-1 to 0.
        rewards = np.array(rewards) / self.args.reward_normalizer

        # Compute value.
        values = []
        minibatches = torch.split(torch.stack(batches), 256)
        for minibatch in minibatches:
            values.append(self.critic.forward(minibatch, action=False).squeeze(1))
        values = torch.cat(values)
        discounted_values = values * gamma ** self.args.n

        # Compute the cumulative discounted returns.
        n_step_rewards = np.zeros((1, self.args.n))
        for i in reversed(range(rewards.shape[0])):
            if i + self.args.n >= rewards.shape[0]:
                V_end = 0
            else:
                V_end = discounted_values[i + self.args.n]
            n_step_rewards[0, :-1] = n_step_rewards[0, 1:] * gamma
            n_step_rewards[0, -1] = rewards[i]

            n_step_return = torch.tensor(n_step_rewards.sum(), device=self.device).unsqueeze(0) + V_end 
            returns.append(n_step_return)

        # Normalize returns.
        # returns = torch.stack(returns)
        # mean_return, std_return = returns.mean(), returns.std()
        # returns = (returns - mean_return) / (std_return + self.eps)

        return torch.stack(returns[::-1]).detach().squeeze(1), torch.stack(log_probs), values.squeeze(), cum_rewards

    def plot(self):
        # Save the plot.
        filename = os.path.join('plots', *self.args.weights_path.split('/')[-2:]).replace('.h5', '.png')
        if not os.path.exists(os.path.dirname(filename)): os.makedirs(os.path.dirname(filename))

        # Make error plot with mean, std of rewards.
        data = np.asarray(self.rewards_data)
        plt.errorbar(data[:, 0], data[:, 1], data[:, 2], lw=2.5, elinewidth=1.5,
            ecolor='grey', barsabove=True, capthick=2, capsize=3)
        plt.title('Cumulative Rewards (Mean/Std) Plot for A3C Algorithm')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Cumulative Rewards')
        plt.grid()
        plt.savefig(filename, dpi=300)
        plt.show()


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--policy_lr', dest='policy_lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic_lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=100, help="The value of N in N-step A2C.")
    parser.add_argument('--reward_norm', dest='reward_normalizer', type=float,
                        default=100.0, help='Normalize rewards by.')
    parser.add_argument('--random_seed', dest='random_seed', type=int,
                        default=999, help='Random Seed')
    parser.add_argument('--test_episodes', dest='test_episodes', type=int,
                        default=25, help='Number of episodes to test on.')
    parser.add_argument('--save_interval', dest='save_interval', type=int,
                        default=1000, help='Weights save interval.')
    parser.add_argument('--test_interval', dest='test_interval', type=int,
                        default=250, help='Test interval.')
    parser.add_argument('--log_interval', dest='log_interval', type=int,
                        default=25, help='Log interval.')
    parser.add_argument('--weights_path', dest='weights_path', type=str,
                        default=None, help='Pretrained weights file.')
    parser.add_argument('--det_eval', action="store_true", default=False,                    
                        help='Deterministic policy for testing.')
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main():
    # Parse command-line arguments.
    args = parse_arguments()
    
    # Create the environment.
    actor_critic = A3C(args, env='Breakout-v0')
    if not args.render: actor_critic.train()


if __name__ == '__main__':
    main()
