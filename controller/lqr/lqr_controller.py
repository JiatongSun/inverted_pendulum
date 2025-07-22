# /controller/lqr/lqr_controller.py

import numpy as np
import scipy.linalg
from controller.controller_base import Controller

class LQRController(Controller):
    def __init__(self, m_pendulum: float, m_cart: float, length: float, g: float, dt: float,
                 u_min: float, u_max: float, x_ref: np.ndarray,
                 Q: np.ndarray = None, R: np.ndarray = None):
        """
        Initializes the LQR controller with system parameters and cost matrices.
        :param m_pendulum: Mass of the pendulum (kg)
        :param m_cart: Mass of the cart (kg)
        :param length: Length of the pendulum (m)
        :param g: Gravitational acceleration (m/s^2)
        :param dt: Time step for discretization (seconds)
        :param u_min: Minimum action/force allowed (N)
        :param u_max: Maximum action/force allowed (N)
        :param x_ref: Reference/desired state [x, theta, x_dot, theta_dot]
        :param Q: State cost matrix (optional, default set in LQR)
        :param R: Control input cost matrix (optional, default set in LQR)
        """
        # Initialize the base class with common parameters
        super().__init__(m_pendulum, m_cart, length, g, dt, u_min, u_max, x_ref)

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
        
        # Default cost matrices if none are provided
        if Q is None:
            Q = np.diag([100, 100, 5, 5])  # Weighting on the states
        if R is None:
            R = np.array([[0.1]])        # Weighting on the control input
        
        self.Q = Q
        self.R = R
        
        # Solve the discrete-time Riccati equation to find the optimal gain matrix K
        self.P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.Bd.T @ self.P @ self.Bd) @ (self.Bd.T @ self.P @ self.Ad)
    
    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the control input using LQR
        :param state: Current state of the system [x, theta, x_dot, theta_dot] (np.ndarray)
        :return: Control input (np.ndarray)
        """
        return -self.K @ state
