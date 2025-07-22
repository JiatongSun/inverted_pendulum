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
    if control_method == "lqr":
        controller = LQRController(m_pendulum, m_cart, length, g, timestep, u_min, u_max, x_ref)
    elif control_method == "mpc":
        controller = MPCController(m_pendulum, m_cart, length, g, timestep, u_min, u_max, x_ref)
    else:
        raise NotImplementedError(f"Control method '{control_method}' is not implemented")


    # Reset env and start simulation
    observation, _ = env.reset()
    for idx in range(1000):
        u = controller.compute_control(observation)
        action = np.clip(u, u_min, u_max)
        observation, reward, terminated, _, _ = env.step(action)

        env.render()

        if terminated:
            print(f"Episode finished at step {idx}, resetting environment.")
            break

    env.close()

if __name__ == "__main__":
    run_pendulum_env(control_method="lqr")
