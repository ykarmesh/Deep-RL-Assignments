import gym 
import torch 
import numpy as np 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 

class ActorNetwork(torch.nn.Module):
    '''This class essentially defines the actor network architecture'''
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(ActorNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        #self.output = nn.Linear(hidden_size, output_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x= torch.tanh(self.linear3(x))
        return x
