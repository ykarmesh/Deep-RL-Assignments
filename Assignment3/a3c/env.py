import os
import sys

ROS_CV = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_CV in sys.path: sys.path.remove(ROS_CV)

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class Worker(mp.Process):
    def __init__(self, env, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = gym.make(env).unwrapped

    def run(self):
        total_step = 1

        states = []
        rewards, returns = [], []
        actions, log_probs = [], []
        while self.g_ep.value < MAX_EP:
            state = self.env.reset()
            states, actions, rewards = [], [], []
            epside_return = 0.
            while True:
                if self.name == 'w0':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


def preprocess(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (84, 84))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.expand_dims(frame, 0)
    return frame


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    env.reset()
    for i in range(50):
        state, _, done, _ = env.step(1)
        state = preprocess(state)
        print(state.shape)
        if done: break
