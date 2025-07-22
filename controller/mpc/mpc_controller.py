# /controller/mpc/mpc_controller.py

import numpy as np
import scipy.linalg
import cvxpy as cp
from controller.controller_base import Controller

class MPCController(Controller):
    def __init__(self, m_pendulum: float, m_cart: float, length: float, g: float, dt: float, 
                 u_min: float, u_max: float, x_ref: np.ndarray, 
                 N: int=10, Q: np.ndarray = None, R: np.ndarray = None):
        """
        Initializes the MPC controller with system parameters and hyperparameters.
        :param m_pendulum: Mass of the pendulum (kg)
        :param m_cart: Mass of the cart (kg)
        :param length: Length of the pendulum (m)
        :param g: Gravitational acceleration (m/s^2)
        :param dt: Time step for discretization (seconds)
        :param u_min: Minimum action/force allowed (N)
        :param u_max: Maximum action/force allowed (N)
        :param x_ref: Reference/desired state [x, theta, x_dot, theta_dot]
        :param N: Number of prediction steps
        :param Q: State cost matrix (optional)
        :param R: Control input cost matrix (optional)
        """
        # Initialize the base class with common parameters
        super().__init__(m_pendulum, m_cart, length, g, dt, u_min, u_max, x_ref)

        self.N = N  # Prediction horizon

        # Default cost matrices if none are provided
        if Q is None:
            Q = np.diag([100, 100, 5, 5])  # Weighting on the states
        if R is None:
            R = np.array([[0.1]])  # Weighting on the control input

        self.Q = Q
        self.R = R

        # Linearized system matrices (A, B) for the pendulum in the upright position
        self.A = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, -self.m_pendulum * self.g / self.m_cart, 0, 0],
                           [0, (self.m_pendulum + self.m_cart) * self.g / (self.m_cart * self.length), 0, 0]])

        self.B = np.array([[0],
                           [0],
                           [1 / self.m_cart],
                           [-1 / (self.m_cart * self.length)]])
        
        # Perform discretization by Tustin transformation
        I = np.eye(self.A.shape[0])
        self.Ad = np.linalg.inv(I - 0.5 * dt * self.A) @ (I + 0.5 * dt * self.A)
        self.Bd = np.linalg.inv(I - 0.5 * dt * self.A) @ (dt * self.B)
        
        # Calculate terminal cost matrix Qf using LQR's discrete-time Riccati equation
        self.Qf = self.compute_terminal_cost_matrix(self.Ad, self.Bd, self.Q, self.R)

        # Define the optimization variables
        self.u = cp.Variable((self.N, 1))  # Control inputs over the prediction horizon
        self.x = cp.Variable((self.N+1, 4))  # States over the prediction horizon (including the initial state)

    def compute_terminal_cost_matrix(self, Ad, Bd, Q, R):
        """
        Solves the discrete-time Riccati equation to compute the terminal cost matrix Qf.
        :param Ad: State matrix (system dynamics)
        :param Bd: Control matrix
        :param Q: State cost matrix
        :param R: Control input cost matrix
        :return: Terminal cost matrix Qf
        """
        # Solve the discrete-time Riccati equation using scipy's solve_discrete_are
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
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
            cost += cp.quad_form(self.x[t, :] - self.x_ref, self.Q)  # State error cost
            cost += cp.quad_form(self.u[t], self.R)  # Control input cost
            # Model dynamics: x_{t+1} = Ad * x_t + Bd * u_t (discrete-time model)
            constraints += [self.x[t+1, :] == self.Ad @ self.x[t, :] + self.Bd @ self.u[t]]
            # Control input bounds: u_min <= u_t <= u_max
            constraints += [self.u[t] <= self.u_max, self.u[t] >= self.u_min]
        
        cost += cp.quad_form(self.x[self.N, :] - self.x_ref, self.Qf)  # Terminal state error cost

        # Define the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        # Solve the optimization problem
        problem.solve(solver='OSQP', max_iter=100000, verbose=False)

        # Return the first control input
        return self.u[0].value