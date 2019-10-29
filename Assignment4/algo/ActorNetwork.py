import copy
import math
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300


class Actor(nn.Module):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    """
    def __init__(self, state_size, action_size, custom_init):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, HIDDEN1_UNITS)

        self.linear2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)

        self.output = nn.Linear(HIDDEN2_UNITS, action_size)

        if custom_init:
            nn.init.uniform_(self.linear1.weight, a=-1/math.sqrt(state_size), b=1/math.sqrt(state_size))
            nn.init.uniform_(self.linear1.bias, a=-1/math.sqrt(state_size), b=1/math.sqrt(state_size))
            nn.init.uniform_(self.linear2.weight, a=-1/math.sqrt(HIDDEN1_UNITS), b=1/math.sqrt(HIDDEN1_UNITS))
            nn.init.uniform_(self.linear2.bias, a=-1/math.sqrt(HIDDEN1_UNITS), b=1/math.sqrt(HIDDEN1_UNITS))
            nn.init.uniform_(self.output.weight, a=-3*10e-3, b=3*10e-3)
            nn.init.uniform_(self.output.bias, a=-3*10e-3, b=3*10e-3)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.output(x))
        return x


class ActorNetwork():
    def __init__(self, state_size, action_size, batch_size,
                 tau, learning_rate, device, custom_init):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.lr = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.policy = Actor(state_size, action_size, custom_init).to(device)
        # self.policy.apply(self.initialize_weights)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy_target = copy.deepcopy(self.policy)


    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.ones_(layer.bias)

    def train(self, action_grads):
        """Updates the actor by applying dQ(s, a) / da.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
            action_grads: a batched numpy array storing the
                gradients dQ(s, a) / da.
        """
        action_loss = -torch.mean(action_grads)
        self.policy_optimizer.zero_grad()
        action_loss.backward()
        self.policy_optimizer.step()

        return action_loss

    def update_target(self):
        """Updates the target net using an update rate of tau."""

        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)