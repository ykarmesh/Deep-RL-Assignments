#!/usr/bin/env python

import os
import pdb
import sys
import copy
import json
import argparse
from datetime import datetime

import gym
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# TensorFlow log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class QNetwork():
    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, args, input, output, learning_rate):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        self.weights_path = 'models/%s/%s' % (args.env, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if args.model_file is None:
            # Network architecture.
            self.model = keras.models.Sequential()      
            self.model.add(Dense(128, activation='relu', input_dim=input, kernel_initializer=keras.initializers.VarianceScaling(scale=2.0)))
            self.model.add(Dense(128, activation='relu', kernel_initializer=keras.initializers.VarianceScaling(scale=2.0)))
            self.model.add(Dense(128, activation='relu', kernel_initializer=keras.initializers.VarianceScaling(scale=2.0)))
            self.model.add(Dense(output, activation='linear', kernel_initializer=keras.initializers.VarianceScaling(scale=2.0)))

            # Loss and optimizer.
            adam = keras.optimizers.Adam(lr=learning_rate)
            self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
        else:
            print('Loading pretrained model from', args.model_file)
            self.load_model_weights(args.model_file)

    def save_model_weights(self, step):
        # Helper function to save your model / weights.
        if not os.path.exists(self.weights_path): os.makedirs(self.weights_path)
        self.model.save(os.path.join(self.weights_path, 'model_%d.h5' % step))

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights. 
        self.model = keras.models.load_model(weight_file)


class Replay_Memory():

    def __init__(self, state_dim, action_dim, memory_size=50000, burn_in=10000):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) way to implement the memory is as a list of transitions. 
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.states = np.zeros((self.memory_size, state_dim)) 
        self.next_states = np.zeros((self.memory_size, state_dim))
        self.actions = np.zeros((self.memory_size, 1))
        self.rewards = np.zeros((self.memory_size, 1))
        self.dones = np.zeros((self.memory_size, 1))
        self.ptr = 0
        self.burned_in = False
        self.not_full_yet = True  

    def append(self, states, actions, rewards, next_states, dones):
        self.states[self.ptr] = states
        self.actions[self.ptr, 0] = actions
        self.rewards[self.ptr, 0] = rewards
        self.next_states[self.ptr] = next_states
        self.dones[self.ptr, 0] = dones
        self.ptr += 1

        if self.ptr > self.burn_in:
            self.burned_in = True

        if self.ptr >= self.memory_size:
            self.ptr = 0
            self.not_full_yet = False

    def sample_batch(self, batch_size=32):   
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        if self.not_full_yet:
            idxs = np.random.choice(self.ptr, batch_size, False)
        else:
            idxs = np.random.choice(self.memory_size, batch_size, False)

        states = self.states[idxs]
        next_states = self.next_states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones


class DQN_Agent():
    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, args):
        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 

        # Inputs
        self.args = args
        self.environment_name = self.args.env
        self.render = self.args.render
        self.epsilon = args.epsilon
        self.network_update_freq = args.network_update_freq
        self.log_freq = args.log_freq
        self.test_freq = args.test_freq
        self.save_freq = args.save_freq
        self.learning_rate = self.args.learning_rate

        # Env related variables
        if self.environment_name == 'CartPole-v0':
            self.env = gym.make(self.environment_name)
            self.discount_factor = 0.99
            self.num_episodes = 5000
        elif self.environment_name == 'MountainCar-v0':
            self.env = gym.make(self.environment_name)
            self.discount_factor = 1.00
            self.num_episodes = 10000
        else:
            raise Exception("Unknown Environment")

        # Other Classes
        self.q_network = QNetwork(args, self.env.observation_space.shape[0], self.env.action_space.shape[0], self.learning_rate)
        self.target_q_network = QNetwork(args, self.env.observation_space.shape[0], self.env.action_space.shape[0], self.learning_rate)
        self.memory = Replay_Memory(self.env.observation_space.shape[0], self.env.action_space.shape[0], memory_size=self.args.memory_size)

        # Plotting
        self.rewards = []
        self.td_error = []
        self.batch = list(range(32))

        # Tensorboard
        self.logdir = 'logs/%s/%s' % (self.environment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.summary_writer = SummaryWriter(self.logdir)

        # Save hyperparameters
        with open(self.logdir + '/hyperparameters.json', 'w') as outfile:
            json.dump(vars(self.args), outfile, indent=4)

    def epsilon_greedy_policy(self, q_values, epsilon):
        # Creating epsilon greedy probabilities to sample from.             
        p = np.random.uniform(0, 1)
        if p < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(q_values)

    def greedy_policy(self, q_values):
        return np.argmax(q_values)

    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters.

        # When use replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        self.burn_in_memory()
        for step in range(self.num_episodes):
            # Generate Episodes using Epsilon Greedy Policy and train the Q network.
            self.generate_episode(policy=self.epsilon_greedy_policy, mode='train',
                epsilon=self.epsilon, frameskip=self.args.frameskip)

            # Test the network.
            if step % self.test_freq == 0:
                test_reward, test_error = self.test(episodes=20)
                self.rewards.append([test_reward, step])
                self.td_error.append([test_error, step])
                self.summary_writer.add_scalar('test/reward', test_reward, step)
                self.summary_writer.add_scalar('test/td_error', test_error, step)

            # Update the target network.
            if step % self.network_update_freq == 0:
                self.hard_update()

            # Logging.
            if step % self.log_freq == 0:
                print("Step: {0:05d}/{1:05d}".format(step, self.num_episodes))

            # Save the model.
            if step % self.save_freq == 0:
                self.q_network.save_model_weights(step)

            step += 1
            self.epsilon_decay()

            # Render and save the video with the model.
            if step % int(self.num_episodes / 3) == 0 and self.args.render:
                # test_video(self, self.environment_name, step)
                self.q_network.save_model_weights(step)

        self.summary_writer.export_scalars_to_json(os.path.join(self.logdir, 'all_scalars.json'))
        self.summary_writer.close()

    def train_dqn(self):
        # Sample from the replay buffer.
        state, action, rewards, next_state, done = self.memory.sample_batch(batch_size=32)

        # Compute the target Q-value for the loss.
        _y = rewards + self.discount_factor * np.multiply((1 - done),
            np.amax(self.target_q_network.model.predict_on_batch(next_state), axis=1, keepdims=True))

        # Replace the non-optimal actions with the predictions using the
        # old states so that it doesn't contribute to the loss.
        y = self.q_network.model.predict_on_batch(state)
        y[self.batch, action.squeeze().astype(int)] = _y.squeeze()

        # Network Input - S | Output - Q(S,A) | Error - (Y - Q(S,A))^2
        history = self.q_network.model.fit(state, y, epochs=1, batch_size=32, verbose=False)
        loss = history.history['loss'][-1]
        acc = history.history['acc'][-1]
        return loss, acc

    def train_double_dqn(self):
        # Sample from the replay buffer.
        state, action, rewards, next_state, done = self.memory.sample_batch(batch_size=32)

        # Pick the next best action from the Q network.
        next_action = np.argmax(self.q_network.model.predict_on_batch(next_state), axis=1)

        # Compute the target Q-value for the loss.
        _y = rewards + self.discount_factor * np.multiply((1 - done),
            self.target_q_network.model.predict_on_batch(next_state)[self.batch, next_action].reshape(-1, 1))

        # Replace the non-optimal actions with the predictions using the
        # old states so that it doesn't contribute to the loss.
        y = self.q_network.model.predict_on_batch(state)
        y[self.batch, action.squeeze().astype(int)] = _y.squeeze()

        # Network Input - S | Output - Q(S,A) | Error - (Y - Q(S,A))^2
        history = self.q_network.model.fit(state, y, epochs=1, batch_size=32, verbose=False)
        loss = history.history['loss'][-1]
        acc = history.history['acc'][-1]
        return loss, acc

    def hard_update(self):
        self.target_q_network.model.set_weights(self.q_network.model.get_weights())

    def test(self, model_file=None, episodes=100):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        cum_reward = []
        td_error = []
        for count in range(episodes):
            reward, error = self.generate_episode(policy=self.epsilon_greedy_policy,
                mode='test', epsilon=0.05, frameskip=self.args.frameskip)
            cum_reward.append(reward)
            td_error.append(error)
        cum_reward = np.array(cum_reward)
        td_error = np.array(td_error)
        print("\nTest Rewards: {0} | TD Error: {1:.4f}\n".format(np.mean(cum_reward), np.mean(td_error)))
        return np.mean(cum_reward), np.mean(td_error)

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        while not self.memory.burned_in:
            self.generate_episode(policy=self.epsilon_greedy_policy, mode='burn_in',
                epsilon=self.epsilon, frameskip=self.args.frameskip)
        print("Burn Complete!")

    def generate_episode(self, policy, epsilon, mode='train', frameskip=1):
        """
        Collects one rollout from the policy in an environment.
        """
        done = False
        state = self.env.reset()
        rewards = 0
        q_values = self.q_network.model.predict(state.reshape(1, -1))
        td_error = []
        while not done:
            action = policy(q_values, epsilon)
            i = 0
            while (i < frameskip) and not done:
                next_state, reward, done, info = self.env.step(action)
                rewards += reward
                i += 1
            next_q_values = self.q_network.model.predict(next_state.reshape(1, -1))
            if mode in ['train', 'burn_in'] :
                self.memory.append(state, action, reward, next_state, done)
            else:
                td_error.append(abs(reward + self.discount_factor * (1 - done) * np.max(next_q_values) - q_values))
            if not done:
                state = copy.deepcopy(next_state)
                q_values = copy.deepcopy(next_q_values)

            # Train the network.
            if mode == 'train':
                if self.args.double_dqn: self.train_double_dqn()
                else: self.train_dqn()

        return rewards, np.mean(td_error)

    def plots(self):
        """
        Plots: 
        1) Avg Cummulative Test Reward over 20 Plots
        2) TD Error
        """
        reward, time =  zip(*self.rewards)
        plt.figure(figsize=(8, 3))
        plt.subplot(121)
        plt.title('Cummulative Reward')
        plt.plot(time, reward)
        plt.xlabel('iterations')
        plt.ylabel('rewards')
        plt.legend()
        plt.ylim([0, None])

        loss, time =  zip(*self.td_error)
        plt.subplot(122)
        plt.title('Loss')
        plt.plot(time, loss)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.show()

    def epsilon_decay(self, initial_eps=1.0, final_eps=0.05):
        if(self.epsilon > final_eps):
            factor = (initial_eps - final_eps) / 10000
            self.epsilon -= factor


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env_name, episodes):
    # Usage: 
    #   you can pass the arguments within agent.train() as:
    #       if episode % int(self.num_episodes/3) == 0:
    #           test_video(self, self.environment_name, episode)
    save_path = "%s/video-%s" % (env_name, episodes)
    if not os.path.exists(save_path): os.makedirs(save_path)

    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    print("Video recording the agent with epsilon {0:.4f}".format(agent.epsilon))
    while not done:
        q_values = agent.q_network.model.predict(state.reshape(1, -1))
        action = agent.greedy_policy(q_values)
        i = 0
        while (i < agent.args.frameskip) and not done:
            env.render()
            next_state, reward, done, info = env.step(action)
            reward_total.append(reward)
            i += 1
        state = next_state
    print("reward_total: {}".format(np.sum(reward_total)))
    env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', action="store_true", default=False)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--double_dqn', dest='double_dqn', type=int, default=0)
    parser.add_argument('--frameskip', dest='frameskip', type=int, default=1)
    parser.add_argument('--update_freq', dest='network_update_freq', type=int, default=10)
    parser.add_argument('--log_freq', dest='log_freq', type=int, default=25)
    parser.add_argument('--test_freq', dest='test_freq', type=int, default=100)
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=500)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=50000)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=1.0)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 
    q_agent = DQN_Agent(args)

    # Render output videos using the model loaded from file.
    if args.render: test_video(q_agent, args.env, step)
    else: q_agent.train()  # Train the model.


if __name__ == '__main__':
    main()
