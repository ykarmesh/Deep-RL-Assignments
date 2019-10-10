import os
import sys
import json
import argparse
from datetime import datetime

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    '''This class essentially defines the network architecture'''
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        # self.linear1_bn = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear2_bn = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear3_bn = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))  # self.linear1_bn()
        x = F.relu(self.linear2(x))  # self.linear2_bn()
        x = F.relu(self.linear3(x))  # self.linear3_bn()
        x = F.log_softmax(self.output(x), dim=1)
        return x


class Reinforce():
    '''Implementation of the policy gradient method REINFORCE.'''
    def __init__(self, args, env, train=True):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the environment.
        self.env = gym.make(env)
        self.environment_name = env

        # Setup model.
        self.model = Model(input_dim=self.env.observation_space.shape[0],
            output_dim=self.env.action_space.n)
        self.model.apply(self.initialize_weights)

        # Setup optimizer.
        self.eps = 1e-10  # To avoid divide-by-zero error.
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # Model weights path.
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.weights_path = 'models/%s/%s' % (self.environment_name, self.timestamp)

        # Load pretrained weights.
        if args.weights_path: self.load_model()
        self.model.to(self.device)

        # Data for plotting.
        self.rewards_data = []  # n * [epoch, mean(returns), std(returns)]

        # Video render mode.
        if args.render:
            self.model.eval()
            self.generate_episode(render=True)
            self.plot()
            return

        # Network training mode.
        if train:
            # Tensorboard logging.
            self.logdir = 'logs/%s/%s' % (self.environment_name, self.timestamp)
            self.summary_writer = SummaryWriter(self.logdir)

            # Save hyperparameters.
            with open(self.logdir + '/training_parameters.json', 'w') as f:
                json.dump(vars(self.args), f, indent=4)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def save_model(self, epoch):
        '''Helper function to save model state and weights.'''
        if not os.path.exists(self.weights_path): os.makedirs(self.weights_path)
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'rewards_data': self.rewards_data,
                    'epoch': epoch},
                    os.path.join(self.weights_path, 'model_%d.h5' % epoch))

    def load_model(self):
        '''Helper function to load model state and weights. '''
        if os.path.isfile(self.args.weights_path):
            print('=> Loading checkpoint', self.args.weights_path)
            self.checkpoint = torch.load(self.args.weights_path)
            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.rewards_data = self.checkpoint['rewards_data']
        else:
            raise Exception('No checkpoint found at %s' % self.args.weights_path)

    def train(self):
        '''Trains the model on a single episode using REINFORCE.'''
        for epoch in range(self.args.num_episodes):
            # Generate epsiode data.
            returns, log_probs = self.generate_episode()

            # Compute loss and policy gradient.
            self.model.train()
            self.optimizer.zero_grad()
            loss = (returns * -log_probs).sum()
            loss.backward()
            self.optimizer.step()

            # Test the model.
            if epoch % self.args.test_interval == 0:
                self.test(epoch)

            # Logging.
            if epoch % self.args.log_interval == 0:
                print('Epoch: {0:05d}/{1:05d} | Loss: {2:.3f}'.format(epoch, self.args.num_episodes, loss))
                self.summary_writer.add_scalar('train/loss', loss, epoch)

            # Save the model.
            if epoch % self.args.save_interval == 0:
                self.save_model(epoch)

        self.summary_writer.close()
        self.plot()

    def test(self, epoch):
        print('\nTesting')
        self.model.eval()
        with torch.no_grad():
            rewards = [self.generate_episode(test=True) for epoch in range(self.args.test_episodes)]
            rewards_mean, rewards_std = np.mean(rewards), np.std(rewards)
            print('Test Rewards (Mean): %.3f | Test Rewards (Std): %.3f\n' % (rewards_mean, rewards_std))
            self.rewards_data.append([epoch, rewards_mean, rewards_std])
            self.summary_writer.add_scalar('test/rewards_mean', rewards_mean, epoch)
            self.summary_writer.add_scalar('test/rewards_std', rewards_std, epoch)

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

        states = []
        rewards, returns = [], []
        actions, log_probs = [], []

        self.model.eval()
        while not done:
            # Run policy on current state to log probabilities of actions.
            states.append(torch.tensor(state, device=self.device).unsqueeze(0))
            action_probs = self.model.forward(states[-1]).squeeze(0)

            # Sample action from the log probabilities.
            action = torch.argmax(torch.distributions.Multinomial(logits=action_probs).sample())
            actions.append(action)
            log_probs.append(action_probs[action])

            # Run simulation with current action to get new state and reward.
            if render: monitor.render()
            state, reward, done, info = self.env.step(action.cpu().numpy())
            rewards.append(reward)

            # Break if the episode takes too long.
            iters += 1
            if iters > max_iters: break

        # Save video and close rendering.
        if render:
            monitor.close()
            print('\nCumulative Rewards:', np.sum(rewards))
            return

        # Return cumulative rewards for test mode.
        if test: return np.sum(rewards)

        # Flip rewards from T-1 to 0.
        rewards = torch.tensor(rewards[::-1], device=self.device)

        # Compute the cumulative discounted returns.
        cumulative_return = 0
        for current_reward in rewards:
            cumulative_return = (cumulative_return * gamma) + current_reward
            returns.append(cumulative_return)

        # Normalize returns.
        returns = torch.stack(returns)
        mean_return, std_return = returns.mean(), returns.std()
        returns = (returns - mean_return) / (std_return + self.eps)

        return returns, torch.stack(log_probs[::-1])

    def plot(self):
        # Make error plot with mean, std of rewards
        pass


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', dest='num_episodes', type=int,
                        default=50000, help='Number of episodes to train on.')
    parser.add_argument('--test_episodes', dest='test_episodes', type=int,
                        default=100, help='Number of episodes to test on.')
    parser.add_argument('--save_interval', dest='save_interval', type=int,
                        default=2000, help='Weights save interval.')
    parser.add_argument('--test_interval', dest='test_interval', type=int,
                        default=500, help='Test interval.')
    parser.add_argument('--log_interval', dest='log_interval', type=int,
                        default=100, help='Log interval.')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, help='The learning rate.')
    parser.add_argument('--weights_path', dest='weights_path', type=str,
                        default=None, help='Pretrained weights file.')
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render', action='store_true',
                              help='Whether to render the environment.')
    parser_group.add_argument('--no-render', dest='render', action='store_false',
                              help='Whether to render the environment.')
    parser.set_defaults(render=False)

    return parser.parse_args()


def main():
    # Parse command-line arguments.
    args = parse_arguments()

    # Train the model using REINFORCE.
    reinforce = Reinforce(args, env='LunarLander-v2')
    if not args.render: reinforce.train()


if __name__ == '__main__':
    main()
