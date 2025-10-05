import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from dataclasses import dataclass
import logging
from tqdm import tqdm
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GRPOConfig:
    """Configuration class for GRPO algorithm"""
    # Environment parameters
    env_name: str = "CartPole-v1"
    max_episode_length: int = 500
    
    # GRPO specific parameters
    group_size: int = 8  # Number of policies in each group
    num_groups: int = 4  # Number of groups
    group_update_frequency: int = 10  # Update groups every N episodes
    
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training parameters
    num_episodes: int = 1000
    batch_size: int = 64
    num_epochs: int = 4
    
    # Uncertainty parameters
    noise_std: float = 0.1
    transition_noise: float = 0.05
    reward_noise: float = 0.02
    
    # Logging and saving
    log_interval: int = 50
    save_interval: int = 100
    log_dir: str = "logs"
    
    def __post_init__(self):
        """Create log directory if it doesn't exist"""
        os.makedirs(self.log_dir, exist_ok=True)
