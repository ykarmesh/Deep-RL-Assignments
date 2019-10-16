import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_space):
        super(ActorCritic, self).__init__()
        self.feat_size = 64 * 7 * 7
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.policy_fc1 = nn.Linear(self.feat_size, 256)
        self.policy_fc2 = nn.Linear(256, action_space)

        self.critic_fc1 = nn.Linear(self.feat_size, 256)
        self.critic_fc2 = nn.Linear(256, 1)

    def forward(self, x, action=True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.feat_size)
        if action:
            action = F.relu(self.policy_fc1(x))
            action = F.log_softmax(self.policy_fc2(action), dim=1)
            return action
        else:
            value = F.relu(self.critic_fc1(x))
            value = self.critic_fc2(value)
            return value


if __name__ == "__main__":
    actor_critic = ActorCritic(4, 2)
    action, value = actor_critic(torch.zeros((1, 4, 84, 84)))
