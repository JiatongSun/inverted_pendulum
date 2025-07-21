# /controller/mpc/mpc_controller.py

import numpy as np
import scipy.linalg
import cvxpy as cp
from controller.controller_base import Controller

class MPCController(Controller):
    def __init__(self, m_pendulum: float, m_cart: float, length: float, g: float, N=10, Q=None, R=None, xr=np.zeros(4), max_force=3.0, dt=0.02):
        """
        Initializes the MPC controller with system parameters and hyperparameters.
        :param m_pendulum: Mass of the pendulum (kg)
        :param m_cart: Mass of the cart (kg)
        :param length: Length of the pendulum (m)
        :param g: Gravitational acceleration (m/s^2)
        :param N: Number of prediction steps
        :param Q: State cost matrix (optional)
        :param R: Control input cost matrix (optional)
        :param xr: Desired state
        :param max_force: Maximum force applied to the cart
        :param dt: Time step for discretization (seconds)
        """
        # Initialize the base class with common parameters
        super().__init__(m_pendulum, m_cart, length, g)

        self.N = N  # Prediction horizon
        self.dt = dt  # Time step

        # Default cost matrices if none are provided
        if Q is None:
            Q = np.diag([10, 100, 5, 5])  # Weighting on the states
        if R is None:
            R = np.array([[0.1]])  # Weighting on the control input

        self.Q = Q
        self.R = R
        self.xr = xr
        self.max_force = max_force

        # Linearized system matrices (A, B) for the pendulum in the upright position
        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, -self.m_pendulum * self.g / self.m_cart, 0, 0],
                           [0, (self.m_pendulum + self.m_cart) * self.g / (self.m_cart * self.length), 0, 0]])

        self.B = np.array([[0],
                           [0],
                           [1 / self.m_cart],
                           [-1 / (self.m_cart * self.length)]])
        
        # Calculate terminal cost matrix Qf using LQR's continuous-time Riccati equation
        self.Qf = self.compute_terminal_cost_matrix(self.A, self.B, self.Q, self.R)

        # Perform discretization using matrix exponentiation
        self.Ad = scipy.linalg.expm(self.A * self.dt)  # Discrete A matrix
        self.Bd = np.linalg.pinv(self.A) @ (self.Ad - np.eye(self.A.shape[0])) @ self.B  # Discrete B matrix

        # Define the optimization variables
        self.u = cp.Variable((self.N, 1))  # Control inputs over the prediction horizon
        self.x = cp.Variable((self.N+1, 4))  # States over the prediction horizon (including the initial state)

    def compute_terminal_cost_matrix(self, A, B, Q, R):
        """
        Solves the continuous-time Riccati equation to compute the terminal cost matrix Qf.
        :param A: State matrix (system dynamics)
        :param B: Control matrix
        :param Q: State cost matrix
        :param R: Control input cost matrix
        :return: Terminal cost matrix Qf
        """
        # Solve the continuous-time Riccati equation using scipy's solve_continuous_are
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        return P  # Terminal cost matrix Qf = P

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the optimal control input using MPC.
        :param state: Current state of the system [x, theta, x_dot, theta_dot] (np.ndarray)
        :return: Control input (np.ndarray)
        """

        # Define the cost function
        cost = 0
        constraints = [self.x[0, :] == state]  # Initial state constraint
        for t in range(self.N):
            cost += cp.quad_form(self.x[t, :] - self.xr, self.Q)  # State error cost
            cost += cp.quad_form(self.u[t], self.R)  # Control input cost
            # Model dynamics: x_{t+1} = Ad * x_t + Bd * u_t (discrete-time model)
            constraints += [self.x[t+1, :] == self.Ad @ self.x[t, :] + self.Bd @ self.u[t]]
            # Control input bounds: -max_force <= u_t <= max_force
            constraints += [self.u[t] <= self.max_force, self.u[t] >= -self.max_force]
        
        cost += cp.quad_form(self.x[self.N, :] - self.xr, self.Qf)  # Terminal state error cost

        # Define the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        # Solve the optimization problem
        problem.solve(solver='OSQP', verbose=False)

        # Return the first control input
        return self.u[0].value