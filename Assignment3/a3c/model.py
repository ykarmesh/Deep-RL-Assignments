import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()  
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(64*7*7, 256)
        self.action_linear = nn.Linear(256, action_space)

        self.fc2 = nn.Linear(64*7*7, 256)
        self.policy_linear = nn.Linear(256, 1)

        self.apply(initialize_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x1 = F.relu(self.fc1(x.view(-1,64*7*7)))
        x2 = F.relu(self.fc2(x.view(-1,64*7*7)))

        return self.action_linear(x1), F.log_softmax(self.policy_linear(x2), dim=1)

if __name__ == "__main__":
    ac = ActorCritic(4, 2)
    x = torch.zeros((1,4,84,84))
    ac(x)

