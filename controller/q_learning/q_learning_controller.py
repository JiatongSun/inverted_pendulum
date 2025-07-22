import numpy as np
import os
import random
import gymnasium as gym
import pickle
from controller.controller_base import Controller

class QLearningController(Controller):
    def __init__(self, m_pendulum: float, m_cart: float, length: float, g: float, dt: float, 
                 u_min: float, u_max: float, x_ref: np.ndarray, 
                 n_state_bins: int = 10, n_action_bins: int = 30, 
                 epsilon: float = 0.1, alpha: float = 0.1, 
                 gamma: float = 0.99, episodes: int = 1000, model_folder: str = None):
        """
        Initializes the Q-learning controller.
        :param m_pendulum: Mass of the pendulum (kg)
        :param m_cart: Mass of the cart (kg)
        :param length: Length of the pendulum (m)
        :param g: Gravitational acceleration (m/s^2)
        :param dt: Time step for discretization (seconds)
        :param x_ref: Reference/desired state [x, theta, x_dot, theta_dot]
        :param u_min: Minimum action/force allowed (N)
        :param u_max: Maximum action/force allowed (N)
        :param n_state_bins: Number of bins to discretize the state space
        :param n_action_bins: Number of bins to discretize the action space
        :param epsilon: Exploration rate for epsilon-greedy strategy
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param episodes: Number of training episodes
        :param model_folder: Folder to save/load the Q-table model (optional)
        """
        super().__init__(m_pendulum, m_cart, length, g, dt, u_min, u_max, x_ref)
        
        # Learning hyperparameters
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.episodes = episodes  # Number of training episodes
        
        # Discretize state space
        self.n_state_bins = n_state_bins
        self.state_space_bins = [n_state_bins + 1] * len(x_ref)  # For each state variable (x, theta, x_dot, theta_dot)

        # Discretize action space (from u_min to u_max)
        self.n_action_bins = n_action_bins
        self.action_space = np.linspace(u_min, u_max, n_action_bins + 1)  # Discretized action space
        
        # Initialize Q-table
        self.Q = np.zeros([*self.state_space_bins, len(self.action_space)])

        # Determine the model folder dynamically if not provided
        if model_folder is None:
            # Use the script's directory and append the model folder path
            self.model_folder = os.path.join(os.path.dirname(__file__), "model")
        else:
            self.model_folder = model_folder
        
        # Create the model folder if it does not exist
        os.makedirs(self.model_folder, exist_ok=True)
        
        # Full path to save/load the model
        self.model_file = os.path.join(self.model_folder, "q_model.pkl")
        
    def discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretizes the continuous state into discrete indices.
        :param state: Continuous state [x, theta, x_dot, theta_dot]
        :return: Discrete state index tuple
        """
        state_idx = []
        for i, val in enumerate(state):
            bins = np.linspace(self.x_ref[i] - 1.0, self.x_ref[i] + 1.0, self.n_state_bins + 1)
            state_idx.append(np.digitize(val, bins) - 1)
        return tuple(state_idx)
    
    def choose_action_idx(self, state: tuple) -> int:
        """
        Chooses an action index using epsilon-greedy strategy.
        :param state: Discretized state index
        :return: Chosen action index
        """
        if random.uniform(0, 1) < self.epsilon:
            action_idx = random.choice(range(len(self.action_space)))  # Randomly explore
        else:
            action_idx = np.argmax(self.Q[state])  # Exploit the best action
        
        return action_idx
    
    def update_q_value(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Updates the Q-table using the Q-learning update rule.
        :param state: Current state index
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state index
        """
        max_next_q = np.max(self.Q[next_state])  # Max Q value of next state
        self.Q[state][action] += self.alpha * (reward + self.gamma * max_next_q - self.Q[state][action])

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the control input using the Q-learning policy.
        :param state: Current state of the system [x, theta, x_dot, theta_dot]
        :return: Control input (np.ndarray)
        """
        state_idx = self.discretize_state(state)
        action_idx = self.choose_action_idx(state_idx)  # Get action index
        
        # Convert action index to force (e.g., action_idx = 0 -> u_min, 1 -> 0, 2 -> u_max)
        action = self.action_space[action_idx]  # Get the corresponding action from action space
        return np.array([action])  # Return the action as numpy array with shape (1,)
    
    def train(self, env: gym.Env):
        """
        Trains the Q-learning controller using the environment.
        :param env: OpenAI Gym environment (e.g., InvertedPendulum)
        """
        for episode in range(self.episodes):
            state = env.reset()[0]  # Initial state
            state_idx = self.discretize_state(state)  # Discretize the initial state
            print(f"Starting episode {episode + 1}/{self.episodes}")
            
            done = False
            total_reward = 0
            
            while not done:
                action_idx = self.choose_action_idx(state_idx)  # Get action index
                action = np.array([self.action_space[action_idx]])  # Map index to the actual action (force)
                next_state, reward, terminated, truncated, _ = env.step(action)  # Perform the action
                
                next_state_idx = self.discretize_state(next_state)  # Discretize next state

                # Update Q-table
                self.update_q_value(state_idx, action_idx, reward, next_state_idx)
                
                state_idx = next_state_idx  # Move to next state
                total_reward += reward

                # Check for termination or truncation to break the episode early
                if terminated or truncated:
                    done = True  # End the episode early
            
            print(f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}")
    
    def save_model(self):
        """
        Saves the Q-table model to the specified file.
        """
        with open(self.model_file, 'wb') as file:
            pickle.dump(self.Q, file)
            print(f"Model saved to {self.model_file}")
    
    def load_model(self):
        """
        Loads the Q-table model from the specified file.
        """
        try:
            with open(self.model_file, 'rb') as file:
                self.Q = pickle.load(file)
                print(f"Model loaded from {self.model_file}")
        except FileNotFoundError:
            print(f"Model file {self.model_file} not found. Proceeding with untrained model.")
