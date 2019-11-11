# import tensorflow as tf
# from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
# from keras.models import Model
# from keras.regularizers import l2
# import keras.backend as K

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

class Model(nn.Module):
    """Creates an critic network.
    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    """
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(state_size + action_size, HIDDEN1_UNITS)
        self.linear2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.linear3 = nn.Linear(HIDDEN2_UNITS, HIDDEN3_UNITS)
        self.output = nn.Linear(HIDDEN3_UNITS, action_size*2)


    def forward(self, x, action):
        x = torch.cat((x, action), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        
        x = self.output(x)
        return x

class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """
        self.device = device
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Log variance bounds
        self.max_logvar = -3*torch.ones((1, self.state_dim), device=self.device).float()
        self.min_logvar = -7*torch.ones((1, self.state_dim), device=self.device).float()

        # TODO write your code here
        # Create and initialize your model
        self.models = []
        self.optimizers = []
        for i in range(self.num_nets):
          self.models.append(Model(self.state_dim, self.action_dim).to(self.device))
          self.optimizers.append(optim.Adam(self.models[-1].parameters(), lr=learning_rate, weight_decay=1e-4))

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        # Bounds logvariance from the top
        logvar = self.max_logvar - nn.Softplus(self.max_logvar - raw_v)
        # Bounds logvariance from the bottom
        logvar = self.min_logvar + nn.Softplus(logvar - self.min_logvar)

        return mean, logvar

    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        batches = (inputs)
        for i in range(epochs):
          for j in range(self.num_nets):
            for k in range(batches):
                pass


        raise NotImplementedError

    # TODO: Write any helper functions that you need

"""
    def create_network(self):
        I = Input(shape=[self.state_dim + self.action_dim], name='input')
        h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
        model = Model(input=I, output=O)
        return model
"""

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PENN(5, 3, 2, 1e-4, device)