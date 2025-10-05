"""
GRPO (Group Relative Policy Optimization) Implementation

This module implements the core GRPO algorithm with detailed explanations
and synthetic data generation for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import GRPOConfig
import logging

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """
    Policy network for GRPO algorithm.
    
    This network outputs action probabilities for discrete action spaces.
    The network uses a simple feedforward architecture with ReLU activations.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights using Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            state: Current state tensor of shape (batch_size, state_dim)
            
        Returns:
            Action probabilities of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy distribution.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action, log_probability)
        """
        action_probs = self.forward(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values in GRPO.
    
    This network estimates the expected return (value) for a given state,
    which is used for computing advantages in the GRPO algorithm.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.
        
        Args:
            state: Current state tensor of shape (batch_size, state_dim)
            
        Returns:
            State values of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class GRPOAgent:
    """
    GRPO (Group Relative Policy Optimization) Agent.
    
    This class implements the core GRPO algorithm which maintains multiple
    groups of policies and uses relative performance between groups to
    guide policy updates. This approach helps with exploration and
    robustness in uncertain environments.
    """
    
    def __init__(self, config: GRPOConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize policy and value networks for each group
        self.policy_networks = []
        self.value_networks = []
        self.policy_optimizers = []
        self.value_optimizers = []
        
        # Group performance tracking
        self.group_performances = np.zeros(config.num_groups)
        self.group_episode_counts = np.zeros(config.num_groups)
        
        # Initialize networks and optimizers
        self._initialize_networks()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
        logger.info(f"Initialized GRPO agent with {config.num_groups} groups, "
                   f"{config.group_size} policies per group")
    
    def _initialize_networks(self):
        """Initialize policy and value networks for all groups"""
        for group_idx in range(self.config.num_groups):
            # Create policy network
            policy_net = PolicyNetwork(
                self.state_dim, 
                self.action_dim, 
                hidden_dim=128
            )
            self.policy_networks.append(policy_net)
            
            # Create value network
            value_net = ValueNetwork(
                self.state_dim, 
                hidden_dim=128
            )
            self.value_networks.append(value_net)
            
            # Create optimizers
            policy_optimizer = torch.optim.Adam(
                policy_net.parameters(), 
                lr=self.config.learning_rate
            )
            value_optimizer = torch.optim.Adam(
                value_net.parameters(), 
                lr=self.config.learning_rate
            )
            
            self.policy_optimizers.append(policy_optimizer)
            self.value_optimizers.append(value_optimizer)
    
    def select_group(self) -> int:
        """
        Select a group based on performance-weighted sampling.
        
        Groups with better performance are more likely to be selected,
        but we maintain some exploration by not always selecting the best group.
        
        Returns:
            Selected group index
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        weights = self.group_performances + epsilon
        
        # Normalize weights to create probability distribution
        probabilities = weights / np.sum(weights)
        
        # Sample group index
        group_idx = np.random.choice(
            self.config.num_groups, 
            p=probabilities
        )
        
        return group_idx
    
    def get_action(self, state: np.ndarray, group_idx: Optional[int] = None) -> Tuple[int, int]:
        """
        Get action from a policy network.
        
        Args:
            state: Current state
            group_idx: Specific group to use (if None, selects based on performance)
            
        Returns:
            Tuple of (action, group_idx)
        """
        if group_idx is None:
            group_idx = self.select_group()
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action from selected policy
        with torch.no_grad():
            action, _ = self.policy_networks[group_idx].get_action(state_tensor)
            action = action.item()
        
        return action, group_idx
    
    def compute_gae_advantages(self, rewards: List[float], values: List[float], 
                             next_values: List[float]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE) advantages.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_values: List of next state value estimates
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if t < len(next_values) else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.lambda_gae * gae
            advantages.insert(0, gae)
            
            # Compute returns
            return_val = rewards[t] + self.config.gamma * next_value
            returns.insert(0, return_val)
        
        return advantages, returns
    
    def update_policy(self, group_idx: int, states: List[np.ndarray], 
                     actions: List[int], advantages: List[float], 
                     returns: List[float], old_log_probs: List[float]):
        """
        Update policy network using GRPO loss.
        
        Args:
            group_idx: Index of the group to update
            states: List of states
            actions: List of actions taken
            advantages: List of computed advantages
            returns: List of computed returns
            old_log_probs: List of old log probabilities
        """
        policy_net = self.policy_networks[group_idx]
        value_net = self.value_networks[group_idx]
        policy_optimizer = self.policy_optimizers[group_idx]
        value_optimizer = self.value_optimizers[group_idx]
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Update policy and value networks
        for epoch in range(self.config.num_epochs):
            # Get current policy outputs
            action_probs = policy_net(states_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            log_probs = action_dist.log_prob(actions_tensor)
            
            # Compute policy loss (GRPO with clipping)
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute entropy loss
            entropy_loss = -action_dist.entropy().mean()
            
            # Total policy loss
            total_policy_loss = policy_loss - self.config.entropy_coef * entropy_loss
            
            # Update policy
            policy_optimizer.zero_grad()
            total_policy_loss.backward()
            policy_optimizer.step()
            
            # Compute value loss
            values = value_net(states_tensor).squeeze()
            value_loss = F.mse_loss(values, returns_tensor)
            
            # Update value network
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
        
        # Store training loss
        self.training_losses.append({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'group_idx': group_idx
        })
    
    def update_group_performance(self, group_idx: int, episode_reward: float):
        """
        Update group performance tracking.
        
        Args:
            group_idx: Index of the group
            episode_reward: Reward obtained in the episode
        """
        # Update running average of group performance
        self.group_episode_counts[group_idx] += 1
        alpha = 0.1  # Learning rate for performance update
        self.group_performances[group_idx] = (
            (1 - alpha) * self.group_performances[group_idx] + 
            alpha * episode_reward
        )
    
    def save_models(self, filepath: str):
        """Save all model parameters"""
        save_dict = {
            'policy_networks': [net.state_dict() for net in self.policy_networks],
            'value_networks': [net.state_dict() for net in self.value_networks],
            'group_performances': self.group_performances,
            'config': self.config
        }
        torch.save(save_dict, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load model parameters"""
        save_dict = torch.load(filepath)
        
        for i, state_dict in enumerate(save_dict['policy_networks']):
            self.policy_networks[i].load_state_dict(state_dict)
        
        for i, state_dict in enumerate(save_dict['value_networks']):
            self.value_networks[i].load_state_dict(state_dict)
        
        self.group_performances = save_dict['group_performances']
        logger.info(f"Models loaded from {filepath}")
