import argparse
import os
import json
import sys
import pdb
from datetime import datetime

from collections import deque
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

from reinforce import Model

class Critic(torch.nn.Module):
    '''This class essentially defines the network architecture'''
    def __init__(self, input_dim, output_dim, hidden_size=16):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        # self.linear1_bn = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear2_bn = nn.BatchNorm1d(hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear3_bn = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.output(x)
        return x

class A2C():
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, args, env, train=True, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        # Create the environment.
        self.env = gym.make(env)
        self.environment_name = env

        # Setup model.
        self.policy = Model(input_dim=self.env.observation_space.shape[0],
                            output_dim=self.env.action_space.n,
                            hidden_size=64)
        self.policy.apply(self.initialize_weights)
        # Setup critic model.
        self.critic = Critic(input_dim=self.env.observation_space.shape[0],
                             output_dim=1, 
                             hidden_size=64)
        self.critic.apply(self.initialize_weights)

        # Setup optimizer.
        self.eps = 1e-10  # To avoid divide-by-zero error.
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)


        # Model weights path.
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.weights_path = 'models/%s/%s' % (self.environment_name, self.timestamp)

        # Load pretrained weights.
        if args.weights_path: self.load_model()
        self.policy.to(self.device)
        self.critic.to(self.device)

        # Data for plotting.
        self.rewards_data = []  # n * [epoch, mean(returns), std(returns)]

        # Video render mode.
        if args.render:
            self.policy.eval()
            self.generate_episode(render=True)
            return

        # Network training mode.
        if train:
            # Tensorboard logging.
            self.logdir = 'logs/%s/%s' % (self.environment_name, self.timestamp)
            self.summary_writer = SummaryWriter(self.logdir)

            # Save hyperparameters.
            with open(self.logdir + '/training_parameters.json', 'w') as f:
                json.dump(vars(self.args), f, indent=4)


        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.


    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

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
            states, returns, log_probs = self.generate_episode()
            value_function = self.critic.forward(states).squeeze(1)
            # Compute loss and policy gradient.
            self.policy_optimizer.zero_grad()
            policy_loss = ((returns - value_function.detach())*(-log_probs)).mean()  # Element wise multiplication.
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
                print('Epoch: {0:05d}/{1:05d} | Policy_Loss: {2:.3f} | Value_Loss: {3:.3f}'.format(epoch, self.args.num_episodes, policy_loss, critic_loss))
                self.summary_writer.add_scalar('train/policy_loss', policy_loss, epoch)
                self.summary_writer.add_scalar('train/critic_loss', critic_loss, epoch)


            # Save the model.
            if epoch % self.args.save_interval == 0:
                self.save_model(epoch)

        self.summary_writer.close()

    def generate_episode(self, gamma=0.99, test=False, render=False, max_iters=2000):
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

        while not done:
            # Run policy on current state to log probabilities of actions.
            states.append(torch.tensor(state, device=self.device).float().unsqueeze(0))
            action_probs = self.policy.forward(states[-1]).squeeze(0)

            # Sample action from the log probabilities.
            if test: action = torch.argmax(action_probs)
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
        if render:
            monitor.close()
            print('\nCumulative Rewards:', np.sum(rewards))
            return

        # Return cumulative rewards for test mode.
        if test: return np.sum(rewards)

        # Flip rewards from T-1 to 0.
        rewards = np.array(rewards) / self.args.reward_normalizer
        # rewards = torch.tensor(rewards, device=self.device).unsqueeze(0)
        # Compute the cumulative discounted returns.
        n_step_rewards = np.zeros((1, self.args.n))
        for i in reversed(range(rewards.shape[0])):
            if i + self.args.n >= rewards.shape[0]:
                V_end = 0
            else:
                V_end = self.critic.forward(states[i + self.args.n]).squeeze(0).detach()          #why is this zero?
            n_step_rewards[0, :-1] = n_step_rewards[0, 1:] * gamma
            n_step_rewards[0, -1] = rewards[i]

            n_step_return = torch.tensor(n_step_rewards.sum(), device=self.device).unsqueeze(0).detach() + V_end * gamma ** self.args.n
            returns.append(n_step_return)

        # Normalize returns.
        # returns = torch.stack(returns)
        # mean_return, std_return = returns.mean(), returns.std()
        # returns = (returns - mean_return) / (std_return + self.eps)

        return torch.stack(states).squeeze(1), torch.stack(returns[::-1]).detach().squeeze(1), torch.stack(log_probs)

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
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--reward_norm', dest='reward_normalizer', type=float,
                        default=100.0, help='Normalize rewards by.')

    parser.add_argument('--test_episodes', dest='test_episodes', type=int,
                        default=100, help='Number of episodes to test` on.')
    parser.add_argument('--save_interval', dest='save_interval', type=int,
                        default=2000, help='Weights save interval.')
    parser.add_argument('--test_interval', dest='test_interval', type=int,
                        default=500, help='Test interval.')
    parser.add_argument('--log_interval', dest='log_interval', type=int,
                        default=50, help='Log interval.')
    parser.add_argument('--weights_path', dest='weights_path', type=str,
                        default=None, help='Pretrained weights file.')

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    
    # Create the environment.
    reinforce = A2C(args, env='LunarLander-v2')
    if not args.render: reinforce.train()
    # TODO: Create the model.

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)
