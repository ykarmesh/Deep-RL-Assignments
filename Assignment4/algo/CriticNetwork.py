import copy
import math
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300


class Critic(nn.Module):
    """Creates an critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    """
    def __init__(self, state_size, action_size, custom_init):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size + action_size, HIDDEN1_UNITS)

        self.linear2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)

        self.output = nn.Linear(HIDDEN2_UNITS, 1)

        if custom_init:
            nn.init.uniform_(self.linear1.weight, a=-1/math.sqrt(state_size+action_size), b=1/math.sqrt(state_size+action_size))
            nn.init.uniform_(self.linear1.bias, a=-1/math.sqrt(state_size+action_size), b=1/math.sqrt(state_size+action_size))
            nn.init.uniform_(self.linear2.weight, a=-1/math.sqrt(HIDDEN1_UNITS), b=1/math.sqrt(HIDDEN1_UNITS))
            nn.init.uniform_(self.linear2.bias, a=-1/math.sqrt(HIDDEN1_UNITS), b=1/math.sqrt(HIDDEN1_UNITS))
            nn.init.uniform_(self.output.weight, a=-3*1e-3, b=3*1e-3)
            nn.init.uniform_(self.output.bias, a=-3*1e-3, b=3*1e-3)

    def forward(self, x, action):
        x = torch.cat((x, action), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        x = self.output(x)
        return x
        


class CriticNetwork(object):
    def __init__(self, state_size, action_size, batch_size,
                 tau, learning_rate, gamma, device, custom_init):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

        Args:
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.lr = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.critic = Critic(state_size, action_size, custom_init).to(device)
        # self.critic.apply(self.initialize_weights)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = copy.deepcopy(self.critic)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.ones_(layer.bias)

    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.
        Note that tf.gradients returns a list storing a single gradient tensor,
        so we return that gradient, rather than the singleton list.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        return self.critic(states, actions)
        
    def train(self, states, actions, rewards, next_states, done, next_actions):
        # Pick the next best action from the Q network.
        with torch.no_grad():
            Q_target = self.critic_target(next_states, next_actions)

            # Compute the target Q-value for the loss.
            y = rewards + self.gamma * (1 - done) * Q_target #check this

        Q_value = self.critic(states, actions)
        # Network Input - S | Output - Q(S,A) | Error - (Y - Q(S,A))^2
        self.critic_optimizer.zero_grad()
        critic_loss =  F.mse_loss(y, Q_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class CriticTD3(nn.Module):
    """Creates an critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    """
    def __init__(self, state_size, action_size, custom_init):
        super(CriticTD3, self).__init__()
        self.linear1 = nn.Linear(state_size + action_size, HIDDEN1_UNITS)
        self.linear2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.output1 = nn.Linear(HIDDEN2_UNITS, 1)

        self.linear3 = nn.Linear(state_size + action_size, HIDDEN1_UNITS)
        self.linear4 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.output2 = nn.Linear(HIDDEN2_UNITS, 1)

        if custom_init:
            nn.init.uniform_(self.linear1.weight, a=-1/math.sqrt(state_size + action_size), b=1/math.sqrt(state_size + action_size))
            nn.init.uniform_(self.linear2.weight, a=-1/math.sqrt(HIDDEN1_UNITS), b=1/math.sqrt(HIDDEN1_UNITS))
            nn.init.uniform_(self.output1.weight, a=-3*1e-3, b=3*1e-3)

            nn.init.uniform_(self.linear3.weight, a=-1/math.sqrt(state_size + action_size), b=1/math.sqrt(state_size + action_size))
            nn.init.uniform_(self.linear4.weight, a=-1/math.sqrt(HIDDEN1_UNITS), b=1/math.sqrt(HIDDEN1_UNITS))
            nn.init.uniform_(self.output2.weight, a=-3*1e-3, b=3*1e-3)

    def forward(self, x, action):

        x = torch.cat((x, action), 1)
        
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))

        x1 = self.output1(x1)

        x2 = F.relu(self.linear3(x))
        x2 = F.relu(self.linear4(x2))

        x2 = self.output2(x2)
        return x1, x2

    def get_Q(self, x, action):

        x = torch.cat((x, action), 1)
        
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))

        x1 = self.output1(x1)

        return x1

class CriticNetworkTD3(object):
    def __init__(self, state_size, action_size, batch_size,
                 tau, learning_rate, gamma, device, custom_init):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

        Args:
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.lr = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.critic = CriticTD3(state_size, action_size, custom_init).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = copy.deepcopy(self.critic)

    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.
        Note that tf.gradients returns a list storing a single gradient tensor,
        so we return that gradient, rather than the singleton list.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        return self.critic.get_Q(states, actions)
        
    def train(self, states, actions, rewards, next_states, done, next_actions):
        # Pick the next best action from the Q network.
        with torch.no_grad():
            Q_target_1, Q_target_2 = self.critic_target(next_states, next_actions)

            # Compute the target Q-value for the loss.
            y = rewards + self.gamma * (1 - done) * torch.min(Q_target_1, Q_target_2) #check this

        Q_value_1, Q_value_2 = self.critic(states, actions)
        # Network Input - S | Output - Q(S,A) | Error - (Y - Q(S,A))^2
        self.critic_optimizer.zero_grad()
        critic_loss =  F.mse_loss(y, Q_value_1) + F.mse_loss(y, Q_value_2)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
