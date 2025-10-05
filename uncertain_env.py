"""
Synthetic RL Environment with Real-World Uncertainties

This module creates a synthetic reinforcement learning environment that
incorporates various types of uncertainties commonly found in real-world
applications, making it ideal for testing GRPO's robustness.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class UncertaintyConfig:
    """Configuration for different types of uncertainties"""
    # Observation noise
    observation_noise_std: float = 0.1
    
    # Action noise (execution uncertainty)
    action_noise_std: float = 0.05
    
    # Transition uncertainty (model uncertainty)
    transition_noise_std: float = 0.02
    
    # Reward uncertainty
    reward_noise_std: float = 0.01
    
    # Delayed rewards (common in real-world)
    reward_delay_prob: float = 0.1
    max_reward_delay: int = 5
    
    # Partial observability
    partial_obs_prob: float = 0.05
    
    # Non-stationary environment (changing dynamics)
    non_stationary_prob: float = 0.02
    dynamics_change_magnitude: float = 0.1


class UncertainCartPoleEnv(gym.Env):
    """
    CartPole environment with various real-world uncertainties.
    
    This environment extends the classic CartPole problem by adding:
    1. Observation noise
    2. Action execution uncertainty
    3. Transition model uncertainty
    4. Reward noise and delays
    5. Partial observability
    6. Non-stationary dynamics
    """
    
    def __init__(self, uncertainty_config: UncertaintyConfig = None):
        super().__init__()
        
        self.uncertainty_config = uncertainty_config or UncertaintyConfig()
        
        # Environment parameters (can change due to non-stationarity)
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5
        self.pole_mass_length = self.pole_mass * self.pole_length
        self.force_magnitude = 10.0
        self.tau = 0.02  # seconds between state updates
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # State variables
        self.state = None
        self.steps_beyond_terminated = None
        
        # Uncertainty tracking
        self.delayed_rewards = []  # List of (reward, steps_remaining)
        self.dynamics_changes = 0
        self.episode_step = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
        logger.info("Initialized UncertainCartPole environment")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset state to random initial position
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps_beyond_terminated = None
        self.episode_step = 0
        self.delayed_rewards = []
        
        # Apply non-stationary dynamics changes
        self._apply_non_stationary_changes()
        
        # Get observation with noise
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        if self.steps_beyond_terminated is not None:
            logger.warning("Calling step() after episode termination")
            return self.state, 0.0, True, False, {}
        
        # Apply action noise (execution uncertainty)
        actual_action = self._apply_action_noise(action)
        
        # Execute step with transition uncertainty
        self.state = self._step_dynamics(self.state, actual_action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.episode_step >= 500
        
        # Compute reward with noise and delays
        reward = self._compute_reward()
        reward = self._apply_reward_noise(reward)
        reward = self._handle_delayed_rewards(reward)
        
        # Get observation with noise and partial observability
        observation = self._get_observation()
        
        self.episode_step += 1
        
        # Apply non-stationary changes during episode
        if random.random() < self.uncertainty_config.non_stationary_prob:
            self._apply_non_stationary_changes()
        
        return observation, reward, terminated, truncated, {}
    
    def _step_dynamics(self, state: np.ndarray, action: int) -> np.ndarray:
        """Step the environment dynamics with transition uncertainty"""
        x, x_dot, theta, theta_dot = state
        
        # Apply force
        force = self.force_magnitude if action == 1 else -self.force_magnitude
        
        # Add transition noise
        force += np.random.normal(0, self.uncertainty_config.transition_noise_std)
        
        # Compute derivatives
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_length * (4.0/3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass
        
        # Update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        x, _, theta, _ = self.state
        return bool(
            x < -2.4 or x > 2.4 or theta < -0.2095 or theta > 0.2095
        )
    
    def _compute_reward(self) -> float:
        """Compute reward for current state"""
        x, x_dot, theta, theta_dot = self.state
        
        # Base reward for staying upright
        reward = 1.0
        
        # Bonus for being centered
        reward += 0.1 * (1.0 - abs(x) / 2.4)
        
        # Penalty for high angular velocity
        reward -= 0.01 * abs(theta_dot)
        
        return reward
    
    def _apply_action_noise(self, action: int) -> int:
        """Apply action execution uncertainty"""
        if random.random() < self.uncertainty_config.action_noise_std:
            return 1 - action  # Flip action with small probability
        return action
    
    def _apply_reward_noise(self, reward: float) -> float:
        """Apply reward noise"""
        noise = np.random.normal(0, self.uncertainty_config.reward_noise_std)
        return reward + noise
    
    def _handle_delayed_rewards(self, reward: float) -> float:
        """Handle delayed rewards (common in real-world scenarios)"""
        # Add current reward to delayed rewards
        if random.random() < self.uncertainty_config.reward_delay_prob:
            delay = random.randint(1, self.uncertainty_config.max_reward_delay)
            self.delayed_rewards.append((reward, delay))
            return 0.0  # No immediate reward
        
        # Process delayed rewards
        immediate_reward = reward
        for i in range(len(self.delayed_rewards) - 1, -1, -1):
            delayed_reward, steps_remaining = self.delayed_rewards[i]
            steps_remaining -= 1
            
            if steps_remaining <= 0:
                immediate_reward += delayed_reward
                self.delayed_rewards.pop(i)
            else:
                self.delayed_rewards[i] = (delayed_reward, steps_remaining)
        
        return immediate_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get observation with noise and partial observability"""
        observation = self.state.copy()
        
        # Apply observation noise
        noise = np.random.normal(0, self.uncertainty_config.observation_noise_std, size=observation.shape)
        observation += noise
        
        # Apply partial observability (mask some observations)
        if random.random() < self.uncertainty_config.partial_obs_prob:
            mask_idx = random.randint(0, len(observation) - 1)
            observation[mask_idx] = 0.0
        
        return observation.astype(np.float32)
    
    def _apply_non_stationary_changes(self):
        """Apply non-stationary changes to environment dynamics"""
        if random.random() < self.uncertainty_config.non_stationary_prob:
            # Change gravity slightly
            self.gravity += np.random.normal(0, self.uncertainty_config.dynamics_change_magnitude)
            self.gravity = max(5.0, min(15.0, self.gravity))  # Keep within reasonable bounds
            
            # Change pole length slightly
            self.pole_length += np.random.normal(0, self.uncertainty_config.dynamics_change_magnitude * 0.1)
            self.pole_length = max(0.3, min(0.7, self.pole_length))
            
            self.pole_mass_length = self.pole_mass * self.pole_length
            self.dynamics_changes += 1
            
            logger.debug(f"Applied non-stationary change: gravity={self.gravity:.3f}, "
                        f"pole_length={self.pole_length:.3f}")
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        if mode == 'human':
            # Simple text-based rendering
            x, x_dot, theta, theta_dot = self.state
            print(f"Step {self.episode_step}: x={x:.3f}, x_dot={x_dot:.3f}, "
                  f"theta={theta:.3f}, theta_dot={theta_dot:.3f}")
    
    def get_uncertainty_stats(self) -> Dict[str, Any]:
        """Get statistics about uncertainties applied"""
        return {
            'dynamics_changes': self.dynamics_changes,
            'delayed_rewards_count': len(self.delayed_rewards),
            'current_gravity': self.gravity,
            'current_pole_length': self.pole_length,
            'episode_step': self.episode_step
        }


class MultiTaskUncertainEnv(gym.Env):
    """
    Multi-task environment with different uncertainty profiles.
    
    This environment switches between different tasks with varying
    uncertainty characteristics, testing the agent's ability to
    adapt to different types of challenges.
    """
    
    def __init__(self, task_configs: List[Dict] = None):
        super().__init__()
        
        # Default task configurations
        if task_configs is None:
            task_configs = [
                {
                    'name': 'low_uncertainty',
                    'uncertainty_config': UncertaintyConfig(
                        observation_noise_std=0.01,
                        action_noise_std=0.01,
                        transition_noise_std=0.005,
                        reward_noise_std=0.005,
                        reward_delay_prob=0.01,
                        partial_obs_prob=0.01,
                        non_stationary_prob=0.005
                    )
                },
                {
                    'name': 'medium_uncertainty',
                    'uncertainty_config': UncertaintyConfig(
                        observation_noise_std=0.05,
                        action_noise_std=0.03,
                        transition_noise_std=0.01,
                        reward_noise_std=0.01,
                        reward_delay_prob=0.05,
                        partial_obs_prob=0.03,
                        non_stationary_prob=0.01
                    )
                },
                {
                    'name': 'high_uncertainty',
                    'uncertainty_config': UncertaintyConfig(
                        observation_noise_std=0.1,
                        action_noise_std=0.05,
                        transition_noise_std=0.02,
                        reward_noise_std=0.02,
                        reward_delay_prob=0.1,
                        partial_obs_prob=0.05,
                        non_stationary_prob=0.02
                    )
                }
            ]
        
        self.task_configs = task_configs
        self.current_task_idx = 0
        self.task_switch_frequency = 100  # Switch tasks every N episodes
        
        # Create base environment
        self.env = UncertainCartPoleEnv(task_configs[0]['uncertainty_config'])
        
        # Inherit spaces from base environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        logger.info(f"Initialized MultiTaskUncertainEnv with {len(task_configs)} tasks")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and potentially switch tasks"""
        # Switch tasks based on frequency
        if hasattr(self, 'episode_count'):
            self.episode_count += 1
            if self.episode_count % self.task_switch_frequency == 0:
                self._switch_task()
        else:
            self.episode_count = 1
        
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step in current task environment"""
        return self.env.step(action)
    
    def _switch_task(self):
        """Switch to a different task"""
        self.current_task_idx = (self.current_task_idx + 1) % len(self.task_configs)
        task_config = self.task_configs[self.current_task_idx]
        
        # Create new environment with different uncertainty profile
        self.env = UncertainCartPoleEnv(task_config['uncertainty_config'])
        
        logger.info(f"Switched to task: {task_config['name']}")
    
    def get_current_task(self) -> str:
        """Get current task name"""
        return self.task_configs[self.current_task_idx]['name']
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        return self.env.render(mode)


def create_uncertain_env(env_type: str = 'cartpole', uncertainty_level: str = 'medium') -> gym.Env:
    """
    Factory function to create uncertain environments.
    
    Args:
        env_type: Type of environment ('cartpole', 'multitask')
        uncertainty_level: Level of uncertainty ('low', 'medium', 'high')
    
    Returns:
        Configured uncertain environment
    """
    uncertainty_configs = {
        'low': UncertaintyConfig(
            observation_noise_std=0.01,
            action_noise_std=0.01,
            transition_noise_std=0.005,
            reward_noise_std=0.005,
            reward_delay_prob=0.01,
            partial_obs_prob=0.01,
            non_stationary_prob=0.005
        ),
        'medium': UncertaintyConfig(
            observation_noise_std=0.05,
            action_noise_std=0.03,
            transition_noise_std=0.01,
            reward_noise_std=0.01,
            reward_delay_prob=0.05,
            partial_obs_prob=0.03,
            non_stationary_prob=0.01
        ),
        'high': UncertaintyConfig(
            observation_noise_std=0.1,
            action_noise_std=0.05,
            transition_noise_std=0.02,
            reward_noise_std=0.02,
            reward_delay_prob=0.1,
            partial_obs_prob=0.05,
            non_stationary_prob=0.02
        )
    }
    
    if env_type == 'cartpole':
        return UncertainCartPoleEnv(uncertainty_configs[uncertainty_level])
    elif env_type == 'multitask':
        return MultiTaskUncertainEnv()
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
