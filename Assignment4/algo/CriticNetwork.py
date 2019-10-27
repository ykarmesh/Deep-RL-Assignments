import copy
import math
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


class Critic(nn.Module):
    """Creates an critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    """
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size, HIDDEN1_UNITS)
        nn.init.uniform_(self.linear1.weight, a=-1/math.sqrt(state_size), b=1/math.sqrt(state_size))

        self.linear2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        nn.init.uniform_(self.linear2.weight, a=-1/math.sqrt(HIDDEN1_UNITS), b=1/math.sqrt(HIDDEN1_UNITS))

        self.action_layer = nn.Linear(action_size, HIDDEN2_UNITS)
        nn.init.uniform_(self.action_layer.weight, a=-1/math.sqrt(action_size), b=1/math.sqrt(action_size))

        self.output = nn.Linear(2*HIDDEN2_UNITS, 1)
        nn.init.uniform_(self.output.weight, a=-3*10e-3, b=3*10e-3)


    def forward(self, x, action):

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        action = F.relu(self.action_layer(action))
        x = torch.cat((x, action), 1)
        x = self.output(x)
        return x
        


class CriticNetwork(object):
    def __init__(self, state_size, action_size, batch_size,
                 tau, learning_rate, gamma, device):
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
        self.critic = Critic(state_size, action_size).to(device)
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
