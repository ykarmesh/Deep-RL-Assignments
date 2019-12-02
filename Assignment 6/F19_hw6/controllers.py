"""LQR, iLQR and MPC."""

import numpy as np
from scipy.linalg import solve_continuous_are


def simulate_dynamics(env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    # Step simulator with perturbed state/action
    env.state = x.copy()
    next_state, _, _, _ = env.step(u, dt)

    # Compute derivative approximation
    diff = next_state - x
    xdot = diff / dt

    return xdot


def approximate_A(env, x, u, dynamics, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    A = np.zeros((x.shape[0], x.shape[0]))

    for i in range(len(x)):
        # Perturb states element-wise
        delta_vector = np.zeros_like(x)
        delta_vector[i] = delta

        # Compute A using finite differences
        A1 = dynamics(env, x + delta_vector, u, dt)
        A2 = dynamics(env, x - delta_vector, u, dt)
        A[:, i] = (A1 - A2) / (2 * delta)

    return A


def approximate_B(env, x, u, dynamics, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]))

    for i in range(len(u)):
        # Perturb actions element-wise
        delta_vector = np.zeros_like(u)
        delta_vector[i] = delta

        # Compute B using finite differences
        B1 = dynamics(env, x, u + delta_vector, dt)
        B2 = dynamics(env, x, u - delta_vector, dt)
        B[:, i] = (B1 - B2) / (2 * delta)

    return B


def calc_lqr_input(env, sim_env, tN=None, max_iter=None):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    # State and action
    x = env.state.copy()
    u = np.zeros(env.action_space.shape[0])

    # Inputs for Ricatti solver
    A = approximate_A(sim_env, x, u, simulate_dynamics)
    B = approximate_B(sim_env, x, u, simulate_dynamics)
    Q = env.Q.copy()
    R = env.R.copy()

    # Solve Ricatti equation and compute gain
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    # Compute optimal control
    u = -K @ (x - env.goal)
    u = np.clip(u, env.action_space.low, env.action_space.high)

    return u