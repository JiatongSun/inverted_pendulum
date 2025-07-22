# /controller/controller_base.py

from abc import ABC, abstractmethod
import numpy as np

class Controller(ABC):
    def __init__(self, m_pendulum: float, m_cart: float, length: float, g: float, dt: float, 
                 u_min: float, u_max: float, x_ref: np.ndarray):
        """
        Initializes the base controller with common system parameters.
        :param m_pendulum: Mass of the pendulum (kg)
        :param m_cart: Mass of the cart (kg)
        :param length: Length of the pendulum (m)
        :param g: Gravitational acceleration (m/s^2)
        :param dt: Time step for discretization (seconds)
        :param x_ref: Reference/desired state [x, theta, x_dot, theta_dot]
        :param u_min: Minimum action/force allowed (N)
        :param u_max: Maximum action/force allowed (N)
        """
        # Common system parameters for all controllers
        self.m_pendulum = m_pendulum
        self.m_cart = m_cart
        self.length = length
        self.g = g
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.x_ref = x_ref

    @abstractmethod
    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the control input using the specific controller logic.
        :param state: Current state of the system [theta, theta_dot, x, x_dot]
        :return: Control input (np.ndarray)
        """
        pass
