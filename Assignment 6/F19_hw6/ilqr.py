"""LQR, iLQR and MPC."""

import sys
import numpy as np
import pdb
import scipy.linalg

from controllers import approximate_A, approximate_B


def simulate_dynamics_next(env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control.
    x: np.array
      The state to test.
    u: np.array
      The command to test.

    Returns
    -------
    next_x: np.array
    """
    # Step simulator with perturbed state/action
    env.state = x.copy()
    next_state, _, _, _ = env.step(u, dt)

    return next_state


def cost_inter(env, x, u):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control.
    x: np.array
      The state to test.
    u: np.array
      The command to test.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    l = u.T @ env.R @ u
    l_x = np.zeros((x.shape[0])) # 2 * (x - env.goal).T @ env.Q 
    l_xx = np.zeros((x.shape[0], x.shape[0])) # 2 * env.Q
    l_u = 2 * u.T @ env.R 
    l_uu = 2 * env.R
    l_ux = np.zeros((u.shape[0], x.shape[0]))

    return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control.
    x: np.array
      The state to test.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    Qf = 1e4 * np.eye(env.observation_space.shape[0])
    l = (x - env.goal).T @ Qf.copy() @ (x - env.goal)
    l_x = 2 * (x - env.goal).T @ Qf.copy()
    l_xx = 2 * Qf.copy()
    
    return l, l_x, l_xx

def get_total_cost(env, X, U, tN):
    l_total = 0
    l, l_x, l_xx = cost_final(env, X[:, -1])
    l_total += l
    for i in range(tN-2,-1,-1):
        # pdb.set_trace()
        l, l_x, l_xx, l_u, l_uu, l_ux = cost_inter(env, X[:, i].copy(), U[:, i].copy()) 
        l_total += l
    return l_total

def simulate(env, x0, U):
    env.state = x0.copy()

    states, rewards = [x0.copy()], []

    count = 0
    while U.shape[1] > count:
        state, reward, done, info = env.step(U[:, count])

        states.append(state)
        rewards.append(reward)

        if done: break
        count += 1
    # pdb.set_trace()
    return np.array(states).T

def forward_pass(env, X, U, k, K):

    # pdb.set_trace()
    env.state = X[:, 0].copy()

    states, rewards, actions = [X[:, 0].copy()], [], []
    last_state = X[:, 0].copy()

    count = 0
    while U.shape[1] > count:
        action = U[:, count] + K[:, :, count] @ (states[count] - last_state) + k[:, count]  # 
        state, reward, done, info = env.step(action)

        states.append(state)
        rewards.append(reward)
        actions.append(action)
        last_state = state.copy()

        if done: break
        count += 1

    return np.array(states).T.copy(), np.array(actions).T.copy(), -np.sum(rewards).copy()   # costs is -ve of reward



def calc_ilqr_input(env, sim_env, old_actions, tN=50, max_iters=1e6, tol=1e-3, lamb_factor=3.0):
    """Calculate the optimal control input for the given state.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_iters: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    x_0 = env.state.copy()
    
    U = old_actions.copy()

    X = simulate(sim_env, x_0, U)
    J_old = sys.float_info.max
    lamb = 1.0
    max_lamb=100000000
    J_store = []

    print("New Iteration")
    for i in range(int(max_iters)):
        k, K = backward_pass(sim_env, X, U, tN, lamb)
        
        # Get control values at control points and new states again by a forward rollout
        X_new, U_new, _ = forward_pass(sim_env, X, U, k, K)

        J_new = get_total_cost(env, X_new, U_new, tN)
        if J_new < J_old:
            J_store.append(J_new)
            # print("Good Iter")
            print("Update Time: {} {}".format(J_new, J_old))
            lamb /= lamb_factor
            X = X_new.copy()
            U = U_new.copy()
            if (abs(J_old - J_new) < tol):
                print("Tolerance reached")
                break
            J_old = J_new

        else:
            # print("Bad Iter with cost: {} {}".format(J_new, J_old))
            lamb *= lamb_factor
            if lamb > max_lamb:
                lamb = max_lamb
                break
            
    print("Returning Actions: iterations {} cost {}".format(i, J_old))
    return U, J_store

def backward_pass(env, X, U, tN=50, lamb=1):   
    # Value function at final timestep is known
    l, l_x, l_xx = cost_final(env, X[:, -1])
    V_x = l_x.copy()
    V_xx = l_xx.copy()
    # Allocate space for feedforward and feeback term
    k = np.zeros((env.action_space.shape[0], tN))
    K = np.zeros((env.action_space.shape[0], env.observation_space.shape[0], tN))
    # Run a backwards pass from N-1 control step
    for i in range(tN-2,-1,-1):
        # pdb.set_trace()
        df_dx = approximate_A(env, X[:, i].copy(), U[:, i].copy(), simulate_dynamics_next)
        df_du = approximate_B(env, X[:, i].copy(), U[:, i].copy(), simulate_dynamics_next)
        l, l_x, l_xx, l_u, l_uu, l_ux = cost_inter(env, X[:, i].copy(), U[:, i].copy()) 

        Q_x = l_x + df_dx.T @ V_x
        Q_u = l_u + df_du.T @ V_x
        Q_xx = l_xx + df_dx.T @ V_xx @ df_dx 
        Q_ux = l_ux + df_du.T @ V_xx @ df_dx
        Q_uu = l_uu + df_du.T @ V_xx @ df_du
        # Q_uu_inv = np.linalg.pinv(Q_uu) 
        
        Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
        Q_uu_evals[Q_uu_evals < 0] = 0.0
        Q_uu_evals += lamb
        Q_uu_inv = Q_uu_evecs @ np.diag(1.0/Q_uu_evals) @ Q_uu_evecs.T

        # Calculate feedforward and feedback terms
        k[:,i] = -Q_uu_inv @ Q_u
        K[:,:,i] = -Q_uu_inv @ Q_ux
        # Update value function for next time step
        V_x = (Q_x - K[:,:,i].T @ Q_uu @ k[:,i]).copy()
        V_xx = (Q_xx - K[:,:,i].T @ Q_uu @ K[:,:,i]).copy()
    
    return k, K