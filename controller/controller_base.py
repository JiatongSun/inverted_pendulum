# /controller/controller_base.py

from abc import ABC, abstractmethod
import numpy as np

class Controller(ABC):
    def __init__(self, m_pendulum: float, m_cart: float, length: float, g: float, dt: float):
        """
        Initializes the base controller with common system parameters.
        :param m_pendulum: Mass of the pendulum (kg)
        :param m_cart: Mass of the cart (kg)
        :param length: Length of the pendulum (m)
        :param g: Gravitational acceleration (m/s^2)
        :param dt: Time step for discretization (seconds)
        """
        # Common system parameters for all controllers
        self.m_pendulum = m_pendulum
        self.m_cart = m_cart
        self.length = length
        self.g = g
        self.dt = dt

    @abstractmethod
    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the control input using the specific controller logic.
        :param state: Current state of the system [theta, theta_dot, x, x_dot]
        :return: Control input (np.ndarray)
        """
        pass
