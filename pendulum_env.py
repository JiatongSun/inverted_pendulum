import gymnasium as gym
import numpy as np
from controller.controller_base import Controller
from controller.lqr.lqr_controller import LQRController
from controller.mpc.mpc_controller import MPCController
from controller.q_learning.q_learning_controller import QLearningController

def create_controller(control_method: str, 
                      m_pendulum: float, m_cart: float, length: float, g: float, dt: float, 
                      u_min: float, u_max: float, x_ref: np.ndarray) -> Controller:
    """
    Creates the appropriate controller based on the selected method.
    :param control_method: Control method to use, can be "lqr", "mpc", or "rl" (str)
    :param m_pendulum: Mass of the pendulum (kg)
    :param m_cart: Mass of the cart (kg)
    :param length: Length of the pendulum (m)
    :param g: Gravitational acceleration (m/s^2)
    :param dt: Simulation timestep
    :param u_min: Minimum action/force allowed (N)
    :param u_max: Maximum action/force allowed (N)
    :param x_ref: Reference/desired state
    :return: The appropriate controller
    """
    if control_method == "lqr":
        return LQRController(m_pendulum, m_cart, length, g, dt, u_min, u_max, x_ref)
    elif control_method == "mpc":
        return MPCController(m_pendulum, m_cart, length, g, dt, u_min, u_max, x_ref)
    elif control_method == "q_learning":
        return QLearningController(m_pendulum, m_cart, length, g, dt, u_min, u_max, x_ref)
    else:
        raise NotImplementedError(f"Control method '{control_method}' is not implemented")

def run_pendulum_env(control_method: str = "lqr", mode: str = "eval") -> None:
    """
    Runs the Inverted Pendulum environment with the selected control method.
    :param control_method: Control method to use, can be "lqr", "mpc", or "rl" (str)
    :param mode: Mode to run, either "train" or "eval"
    """
    env = gym.make('InvertedPendulum-v5', render_mode='human')

    # Simulation timestep
    timestep = env.unwrapped.model.opt.timestep

    # Action range
    u_min = env.action_space.low[0]
    u_max = env.action_space.high[0]

    # Reference state
    x_ref = np.zeros(env.observation_space.shape[0])

    # System parameters for the pendulum and cart
    m_pendulum = 5.02  # Pendulum mass (kg)
    m_cart = 10.47     # Cart mass (kg)
    length = 0.6       # Length of the pendulum (m)
    g = 9.81           # Gravitational acceleration (m/s^2)

    # Select controller
    controller = create_controller(control_method, m_pendulum, m_cart, length, g, timestep, u_min, u_max, x_ref)

    if mode == "train":
        # Train & save the model
        if control_method in ["q_learning"]:
            controller.load_model()
            controller.train(env)  # Train RL controller
            controller.save_model()
        else:
            print("Training for non-learning-based controllers is not implemented in this version.")
    elif mode == "eval":
        # Load the model
        if control_method in ["q_learning"]:
            controller.load_model()

        # Evaluate the trained model
        observation, _ = env.reset()
        for idx in range(1000):
            u = controller.compute_control(observation)
            action = np.clip(u, u_min, u_max)
            observation, reward, terminated, truncated, _ = env.step(action)

            env.render()

            if terminated or truncated:
                print(f"Episode finished at step {idx}, resetting environment.")
                break
    else:
        raise ValueError(f"Mode '{mode}' is not supported. Use 'train' or 'eval'.")

    env.close()

if __name__ == "__main__":
    run_pendulum_env(control_method="q_learning", mode="train")  # Set mode to "train" or "eval"
