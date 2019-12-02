import os
# import tensorflow as tf
import numpy as np
import gym
import copy
import pdb

class RandomOptimizer:
    def __init__(self, MPC): 
        self.MPC = MPC

        self.plan_horizon = MPC.plan_horizon
        self.popsize = MPC.popsize * MPC.max_iters
        self.num_trajectories = self.MPC.num_particles if (not self.MPC.use_gt_dynamics) else 1

    def act(self, mu, sigma, state):
        states = np.tile(state, (self.popsize,1)) #check this
        states = np.tile(states, (self.num_trajectories,1))
        cost = np.zeros((self.popsize,1))
        # 
        actions =  np.clip(np.array([np.random.multivariate_normal(mu[i], sigma[i], self.popsize) for i in range(self.plan_horizon)]), self.MPC.ac_lb, self.MPC.ac_ub)
        for j in range(self.plan_horizon):
            action_this_step = actions[j, :, :].squeeze()
            action_this_step = np.tile(action_this_step, (self.num_trajectories,1))
            next_states = np.array(self.MPC.predict_next_state(states, action_this_step))
            for k in range(self.popsize):
                cost[k%self.popsize] += self.MPC.obs_cost_fn(next_states[k])
            cost = cost/self.num_trajectories
            states = copy.deepcopy(next_states)
        
        best_idx = np.argmin(cost)

        return actions[:, best_idx, :]



class CEMOptimizer:
    def __init__(self, MPC):
        self.MPC = MPC

        self.max_iters = MPC.max_iters
        self.plan_horizon = MPC.plan_horizon
        self.popsize = MPC.popsize
        self.num_elites = MPC.num_elites
        self.num_trajectories = self.MPC.num_particles if (not self.MPC.use_gt_dynamics) else 1

    def act(self, mu, sigma, state): #check this
        for i in range(self.max_iters):
            states = np.tile(state, (self.popsize, 1))
            states = np.tile(states, (self.num_trajectories,1))
            cost = np.zeros((self.popsize,1))
            actions = np.clip(np.array([np.random.multivariate_normal(mu[i], sigma[i], self.popsize) for i in range(self.plan_horizon)]), self.MPC.ac_lb, self.MPC.ac_ub)
            for j in range(self.plan_horizon):
                action_this_step = actions[j, :, :].squeeze()
                action_this_step = np.tile(action_this_step, (self.num_trajectories,1))
                next_states = np.array(self.MPC.predict_next_state(states, action_this_step)) # need to send appropriate action.
                for k in range(len(next_states)):
                    cost[k%self.popsize] += self.MPC.obs_cost_fn(next_states[k])
                cost = cost/self.num_trajectories
                states = copy.deepcopy(next_states)
            mu, _ = self.get_fit_population(cost, actions)
        return mu

    def get_fit_population(self, cost, actions):
        zipped_pairs = zip(cost, np.swapaxes(actions, 0, 1))
        sorted_cost, sorted_actions = [], []
        
        for (c,a) in sorted(zipped_pairs, key=lambda x: x[0]):
            sorted_cost.append(c)
            sorted_actions.append(a)
        # pdb.set_trace()
        sorted_actions = np.array(sorted_actions)
        mu = np.mean(sorted_actions[:self.num_elites], axis=0)

        sigma = np.array([np.cov(sorted_actions[i, :self.num_elites], rowvar=False) for i in range(self.plan_horizon)])
        # sigma = np.cov(sorted_actions[:self.num_elites], rowvar=False) # + self.noise**2 * np.eye()
    
        return mu, sigma



class MPC:
    def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """

        :param env:
        :param plan_horizon:
        :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param num_particles: Number of trajectories for TS1
        :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
        :param use_mpc: Whether to use only the first action of a planned trajectory
        :param use_random_optimizer: Whether to use CEM or take random actions
        """
        self.env = env
        self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
        self.num_particles = num_particles
        self.plan_horizon = plan_horizon
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low



        # Set up optimizer
        self.model = model

        self.predict_next_state = None
        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # TODO: write your code here
        # Initialize your planner with the relevant arguments.
        # Write different optimizers for cem and random actions respectively
        self.reset()
        if self.use_random_optimizer:
            self.policy=RandomOptimizer(self)
        else:
            self.policy=CEMOptimizer(self)

    def obs_cost_fn(self, state):
        """ Cost function of the current state """
        # Weights for different terms
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to predict the next state"""
        next_states = self.model.get_nxt_state(states, actions)
        return next_states

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        next_states = []
        for i in range(len(states)):
            next_states.append(self.env.get_nxt_state(states[i], actions[i]))
        return next_states

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        obs_trajs_inputs = [obs_traj[:-1,:] for obs_traj in obs_trajs]
        obs_trajs_targets = [obs_traj[1:,:8] for obs_traj in obs_trajs]
        acs_trajs = np.concatenate(acs_trajs)

        inputs = np.concatenate(obs_trajs_inputs)
        inputs[:,-2:] = acs_trajs
        targets = np.concatenate(obs_trajs_targets)
        self.model.train(inputs, targets, epochs=epochs)

    def reset(self):
        # TODO: write your code here
        self.mean = np.zeros([self.plan_horizon, self.action_dim])
        self.cov = np.tile(0.5*np.eye(self.action_dim), (self.plan_horizon, 1, 1))
        # self.cov = 0.5*np.ones([self.plan_horizon, self.action_dim])

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        # TODO: write your code here
        if not self.use_mpc:
            if t%self.plan_horizon == 0:
                # self.reset()
                self.planned_actions = self.policy.act(self.mean, self.cov, state)
                i = 0
            else:
                i = t%self.plan_horizon
            return self.planned_actions[i]
        else:
            self.planned_actions = self.policy.act(self.mean, self.cov, state)
            self.mean[:-1] = self.planned_actions[1:]
            self.mean[-1] = np.zeros([1, self.action_dim])
            self.cov = np.tile(0.5*np.eye(self.action_dim), (self.plan_horizon, 1, 1))

            return self.planned_actions[0]


    # TODO: write any helper functions that you need
