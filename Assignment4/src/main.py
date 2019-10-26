import os 
import pdb
import sys 
import json 
import argparse
from datetime import datetime 
import envs

import gym 
import torch 
import numpy as np 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from tensorboardX import SummaryWriter 
import matplotlib.pyplot as plt 

#Actor and critic networks 
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

from ReplayBuffer import ReplayBuffer 


# Adding noise to actions 
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

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)



class DDPG():
    ''' Implementation of Deep deterministic policy gradients'''
    def __init__(self, args, environment_name, train=True):
        self.args= args
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Creating the environment
        self.env= gym.make(environment_name)
        self.environment_name= environment_name

        # Data parameters
        self.num_observations= self.env.observation_space.shape[0]
        #num_actions= self.env.action_space.n
        # We are operating in continuous space, hence, the action will be a single value 
        self.num_actions= 2


        #Setup models
        self.actor= ActorNetwork(input_dim=  self.num_observations,
                                         output_dim= self.num_actions,
                                        hidden_size= self.args.actor_hidden_neurons)

        self.target_actor= ActorNetwork(input_dim=  self.num_observations,
                                                output_dim= self.num_actions,
                                                hidden_size= self.args.actor_hidden_neurons)

        
        self.critic= CriticNetwork(input_dim= self.num_observations + self.num_actions,
                                           output_dim= 1,
                                           hidden_size= self.args.critic_hidden_neurons)
 
        self.target_critic= CriticNetwork(input_dim= self.num_observations + self.num_actions,
                                           output_dim= 1, 
                                           hidden_size= self.args.critic_hidden_neurons)

        self.actor.apply(self.initialize_weights)
        self.critic.apply(self.initialize_weights)

        # Copy the parameters from the actor and critic to target actor and target critic in the start
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data) 
     
        # Initialize replay buffer 
        self.replay_buffer= ReplayBuffer(self.num_observations, self.num_actions, self.device, memory_size=self.args.memory_size, burn_in=10000)
        
        # Set training parameters 
        self.critic_loss_criterion= nn.MSELoss()
        self.actor_optimizer= optim.Adam(self.actor.parameters(),   lr= self.args.actor_lr)
        self.critic_optimizer= optim.Adam(self.critic.parameters(), lr= self.args.critic_lr)

        pass
    

    def update_nets(self):
        states, actions, rewards, next_states, dones= self.replay_buffer.get_batch(self.args.batch_size)

        # Find target output using target nets
        #Qval= self.critic.forward(states,actions).squeeze(0)    

        Qval= self.critic.forward(states,actions)     
        nextQVal= self.target_critic.forward(next_states,  self.target_actor.forward(states))   
        yi= rewards + self.args.gamma*(nextQVal)

        # Critic loss
        critic_loss= self.critic_loss_criterion(yi, Qval)
        # Actor loss
        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        

        print('-------------------------------------------------------------')
        print ('Critic Loss: ', critic_loss)
        print('-------------------------------------------------------------')
        print ('Actor Loss: ', actor_loss)
        print('-------------------------------------------------------------')
        
        # Perform back prop for both actor and critic 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # Update the target critic and actor network by doing tau delay based updates  
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.args.tau*param  + (1-self.args.tau*param)*target_param)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.args.tau*param  + (1-self.args.tau*param)*target_param)


    # Code to the test the trained neural network using new states 
    def test(self):
        pass

    def train(self):
        noise_model= EpsilonNormalActionNoise(self.args.noise_mu, self.args.noise_sigma, self.args.epsilon)
        rewards= []
        mean_rewards= []
        # Going through all the episodes for the network 
        for epi in range(0, self.args.num_episodes):
            state= self.env.reset()
            episode_reward= 0.0
            done= False 
            step= 0
            loss= 0
            states= []
            actions= []

            while not done:
                # Get action by passing state through the neural network 
                state= torch.tensor(state, device=self.device).float().unsqueeze(0)
                states.append(state)
                action = self.actor.forward(states[-1]).squeeze(0)
                actions.append(action)
                # Get noisy action by passing action through the noise model 
                noisy_action= noise_model(action.detach().numpy())
                # Pass action through state transition (env.step)
                #next_state, reward, done, info = self.env.step(action.cpu().numpy()
                next_state, reward, done, _= self.env.step(noisy_action)
                # Get new state, and reward. Append it to the replay buffer 
                self.replay_buffer.add(state, 
                                       torch.tensor(noisy_action, device=self.device).float().unsqueeze(0), 
                                       reward, 
                                       torch.tensor(next_state, device=self.device).float().unsqueeze(0),
                                       done)
                # If the Replay buffer size > batch_size, we update the model 
                if(self.replay_buffer.burned_in):
                    self.update_nets()    
                # Update state variable to new_state
                state= next_state
                # Store episode rewards and mean rewards
                episode_reward= episode_reward+ reward
            
            rewards.append(episode_reward)
                 

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            

    '''
    def save_model(self, epoch):
        Helper function to save model state and weights.
        if not os.path.exists(self.weights_path): os.makedirs(self.weights_path)
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'rewards_data': self.rewards_data,
                    'epoch': epoch},
                    os.path.join(self.weights_path, 'model_%d.h5' % epoch))

    def load_model(self):
        Helper function to load model state and weights. 
        if os.path.isfile(self.args.weights_path):
            print('=> Loading checkpoint', self.args.weights_path)
            self.checkpoint = torch.load(self.args.weights_path)
            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.rewards_data = self.checkpoint['rewards_data']
        else:
            raise Exception('No checkpoint found at %s' % self.args.weights_path)
    '''

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
    parser.add_argument('--actor_lr', dest='actor_lr', type=float,
                        default=5e-4, help='The Actor learning rate.')
    parser.add_argument('--critic_lr', dest='critic_lr', type=float,
                        default=5e-4, help='The Critic learning rate.')
    parser.add_argument('--max_iters', dest='max_iters', type=int,
                        default=10000, help='Trajectory max iterations.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=1024, help='Batch size for utility vector.')
    parser.add_argument('--weights_path', dest='weights_path', type=str,
                        default=None, help='Pretrained weights file.')
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=50000)
    parser.add_argument('--gamma', dest='gamma', type=int, default=0.99)

    parser.add_argument('--mu', dest='noise_mu', type=int, default= 0)
    parser.add_argument('--sigma', dest='noise_sigma', type=int, default= 0.2)
    parser.add_argument('--epsilon', dest='epsilon', type=int, default= 0.02)
    parser.add_argument('--tau', dest='tau', type=int, default= 0.2)



    parser.add_argument('--actor_hidden_neurons', dest='actor_hidden_neurons', type=int,
                        default=64, help='Number of neurons in the hidden layer of actor')
    parser.add_argument('--critic_hidden_neurons', dest='critic_hidden_neurons', type=int,
                        default=64, help='Number of neurons in the hidden layer of critic function')
    
    '''
    parser.add_argument('--det_eval', action="store_true", default=False,
                        help='Deterministic policy for testing')
    '''
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render', action='store_true',
                              help='Whether to render the environment.')
    parser_group.add_argument('--no-render', dest='render', action='store_false',
                              help='Whether to render the environment.')
    parser.set_defaults(render=False)

    return parser.parse_args()

def main():

    #Parse command line args
    args= parse_arguments()

    #Train model using DDPG
    ddpg= DDPG(args, environment_name='Pushing2D-v0')
    
    # No rendering in train mode 
    if not args.render: ddpg.train()


if __name__ == '__main__':
    main()


