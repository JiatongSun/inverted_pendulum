# pendulum_env.py

import gymnasium as gym
import numpy as np
from controller.lqr.lqr_controller import LQRController
from controller.mpc.mpc_controller import MPCController

def run_pendulum_env(control_method: str = "lqr") -> None:
    """
    Runs the Inverted Pendulum environment with the selected control method.
    :param control_method: Control method to use, can be "lqr", "mpc", or "rl" (str)
    """
    env = gym.make('InvertedPendulum-v5', render_mode='human')

    # Simulation timestep
    timestep = env.unwrapped.model.opt.timestep

    # Action space
    min_action = env.action_space.low
    max_action = env.action_space.high

    # System parameters for the pendulum and cart
    m_pendulum = 5.02  # Pendulum mass (kg)
    m_cart = 10.47     # Cart mass (kg)
    length = 0.6       # Length of the pendulum (m)
    g = 9.81           # Gravitational acceleration (m/s^2)

    # Select controller
    if control_method == "lqr":
        controller = LQRController(m_pendulum, m_cart, length, g)
    elif control_method == "mpc":
        controller = MPCController(m_pendulum, m_cart, length, g, dt=timestep)
    else:
        raise NotImplementedError(f"Control method '{control_method}' is not implemented")


    # Reset env and start simulation
    observation, _ = env.reset()
    for idx in range(1000):
        u = controller.compute_control(observation)
        action = np.clip(u, min_action, max_action)
        observation, reward, terminated, _, _ = env.step(action)

        env.render()

        if terminated:
            print(f"Episode finished at step {idx}, resetting environment.")
            break

    env.close()

if __name__ == "__main__":
    run_pendulum_env(control_method="mpc")
