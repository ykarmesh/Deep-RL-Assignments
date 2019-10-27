import torch
import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, memory_size, burn_in, state_dim, action_dim, device):
        self.memory_size = memory_size
        self.device = device

        self.burn_in = burn_in
        self.states = torch.zeros((self.memory_size, state_dim)) 
        self.next_states = torch.zeros((self.memory_size, state_dim))
        self.actions = torch.zeros((self.memory_size, action_dim))
        self.rewards = torch.zeros((self.memory_size, 1))
        self.dones = torch.zeros((self.memory_size, 1))
        self.ptr = 0
        self.burned_in = False
        self.not_full_yet = True  

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        if self.not_full_yet:
            idxs = np.random.choice(self.ptr, batch_size, False)
        else:
            idxs = np.random.choice(self.memory_size, batch_size, False)
        idxs = torch.LongTensor(idxs)

        states = self.states[idxs].to(self.device)
        next_states = self.next_states[idxs].to(self.device)
        actions = self.actions[idxs].to(self.device)
        rewards = self.rewards[idxs].to(self.device)
        dones = self.dones[idxs].to(self.device)
        return states, actions, rewards, next_states, dones

    def size(self):
        return self.memory_size

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr, 0] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr, 0] = done
        self.ptr += 1

        if self.ptr > self.burn_in:
            self.burned_in = True

        if self.ptr >= self.memory_size:
            self.ptr = 0
            self.not_full_yet = False

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        if self.not_full_yet:
            return self.ptr
        else:
            return self.memory_size

