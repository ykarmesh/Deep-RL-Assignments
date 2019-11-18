    # import tensorflow as tf
# from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
# from keras.models import Model
# from keras.regularizers import l2
# import keras.backend as K
import os
import sys
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader, RandomSampler

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
        self.output = nn.Linear(HIDDEN3_UNITS, state_size*2)


    def forward(self, x, action):
        x = torch.cat((x, action), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        
        x = self.output(x)
        return x

class StoredData(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device, summary_writer, timestamp, environment_name, loading_weights_path=None):
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
        self.loading_weights_path = loading_weights_path

        # Log variance bounds
        self.max_logvar = -3*torch.ones((1, self.state_dim), device=self.device).float()
        self.min_logvar = -7*torch.ones((1, self.state_dim), device=self.device).float()

        self.total_epochs = 0

        # TODO write your code here
        # Create and initialize your model
        self.models = []
        self.optimizers = []
        for i in range(self.num_nets):
            self.models.append(Model(self.state_dim, self.action_dim).to(self.device))
            self.optimizers.append(optim.Adam(self.models[-1].parameters(), lr=learning_rate, weight_decay=1e-4))

        self.weights_path = 'models/%s/%s' % (environment_name, timestamp)
        if self.loading_weights_path: self.load_model()
        self.summary_writer = summary_writer
        

    def save_model(self, epoch):
        '''Helper function to save model state and weights.'''
        if not os.path.exists(self.weights_path): os.makedirs(self.weights_path)

        save_dict = {}
        for i in range(self.num_nets):
            save_dict['model_'+str(i)] = self.models[i].state_dict()
            save_dict['opt_'+str(i)] = self.optimizers[i].state_dict()

        save_dict['epoch'] = epoch
        torch.save(save_dict, os.path.join(self.weights_path, 'model_%d.h5' % epoch))

    def load_model(self):
        '''Helper function to load model state and weights. '''
        if os.path.isfile(self.weights_path):
            print('=> Loading checkpoint', self.weights_path)
            self.checkpoint = torch.load(self.weights_path)
            for i in range(self.num_nets):
                self.models[i].load_state_dict(self.checkpoint['model_'+str(i)])
                self.optimizers[i].load_state_dict(self.checkpoint['opt_'+str(i)])
        else:
            raise Exception('No checkpoint found at %s' % self.weights_path)

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
        logvar = self.max_logvar - nn.Softplus()(self.max_logvar - raw_v)
        # Bounds logvariance from the bottom
        logvar = self.min_logvar + nn.Softplus()(logvar - self.min_logvar)

        return mean, logvar

    def sample_next_state(self, state, action):
        model_idx = torch.randint(self.num_nets, [state.shape[0]])
        means = []; logvars=[]
        for i in range(self.num_nets):
            mean, logvar = self.get_output(self.models[i](state, action))
            means.append(mean)
            logvars.append(logvar)
        means = torch.stack(means)
        logvars = torch.stack(logvars)
        mean = means[model_idx, torch.arange(state.shape[0]),:]
        logvar = logvars[model_idx, torch.arange(state.shape[0]),:]
        
        normal_dist = Normal(mean.flatten(), torch.exp(logvar).flatten())
        sample_states = normal_dist.sample().view(state.shape)
        return sample_states
        

    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
            inputs: state and action inputs.  Assumes that inputs are standardized.
            targets: resulting states
        """     
        inputs = torch.tensor(inputs, device=self.device).float()
        targets = torch.tensor(targets, device=self.device).float()
        transition_dataset = StoredData(inputs, targets)
        sampler = RandomSampler(transition_dataset, replacement=True)
        loader = DataLoader(transition_dataset, batch_size=128, sampler=sampler)
        for i in range(epochs):
            total_loss_epochs = []
            total_rmse_epochs = []
            for j in range(self.num_nets):
                total_loss = []
                total_rmse = []
                for k, (x, target) in enumerate(loader):
                    self.optimizers[j].zero_grad()
                    mean, logvar = self.get_output(self.models[j](x[:,:8], x[:,-2:]))
                    error_sq = (mean-target)**2
                    loss = torch.mean(torch.sum(error_sq/torch.exp(logvar),1) + torch.log(torch.prod(torch.exp(logvar), 1)))
                    # loss = torch.t(mean-target)*torch.diag(torch.exp(logvar))*(mean-target) + torch.log(torch.det(torch.exp(logvar)))
                    loss.backward()
                    self.optimizers[j].step()

                    rmse_error = torch.mean(torch.sqrt(torch.mean(error_sq, dim=1)))
                    total_loss.append(loss.detach().cpu().numpy())
                    total_rmse.append(rmse_error.detach().cpu().numpy())
                self.summary_writer.add_scalar('train/loss_'+str(j), np.mean(total_loss, 0), self.total_epochs)
                self.summary_writer.add_scalar('train/rmse_error_'+str(j), np.mean(total_rmse, 0), self.total_epochs)
                total_loss_epochs.append(np.mean(total_loss, 0))
                total_rmse_epochs.append(np.mean(total_rmse, 0))
            self.summary_writer.add_scalar('train/loss', np.mean(total_loss_epochs, 0), self.total_epochs)
            self.summary_writer.add_scalar('train/rmse_error', np.mean(total_rmse_epochs, 0), self.total_epochs)
            self.total_epochs += 1
            print("Trained epoch: ", i)
            if(self.total_epochs%100 == 0):
                self.save_model(self.total_epochs)

    def get_nxt_state(self, state, action):
        """
        Arguments:
            state: the current state
            action: the current action
        """
        # using some random model for now. will think about sampling later
        state = torch.tensor(state[:,:8], device=self.device).float()#.unsqueeze(0)
        action = torch.tensor(action, device=self.device).float()#.unsqueeze(0)

        # try:
        with torch.no_grad():
            next_state = self.sample_next_state(state, action).cpu().numpy().squeeze()
        # except:
        return next_state


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
