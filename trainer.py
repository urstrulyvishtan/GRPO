"""
Training Loop for GRPO Algorithm

This module implements the complete training pipeline for GRPO,
including data collection, policy updates, logging, and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
import os
from datetime import datetime
import pandas as pd

from config import GRPOConfig
from grpo_agent import GRPOAgent
from uncertain_env import create_uncertain_env, UncertainCartPoleEnv
import gymnasium as gym

logger = logging.getLogger(__name__)

class GRPOTrainer:
    """
    Trainer class for GRPO algorithm.
    
    This class handles the complete training pipeline including:
    - Environment interaction
    - Data collection and storage
    - Policy updates using GRPO
    - Logging and visualization
    - Model saving and loading
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        
        # Initialize environment
        self.env = create_uncertain_env('cartpole', 'medium')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Initialize GRPO agent
        self.agent = GRPOAgent(config, self.state_dim, self.action_dim)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.group_performances_history = []
        self.training_losses = []
        
        # Logging setup
        self.setup_logging()
        
        logger.info(f"Initialized GRPO trainer with {config.num_groups} groups")
    
    def setup_logging(self):
        """Setup logging and visualization directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config.log_dir, f"grpo_run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.models_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Setup file logging
        log_file = os.path.join(self.run_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def collect_episode_data(self) -> Dict:
        """
        Collect data from a single episode.
        
        Returns:
            Dictionary containing episode data
        """
        state, _ = self.env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'group_indices': [],
            'done': False
        }
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.max_episode_length):
            # Get action from agent
            action, group_idx = self.agent.get_action(state)
            
            # Get value estimate
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                value = self.agent.value_networks[group_idx](state_tensor).item()
            
            # Get log probability
            with torch.no_grad():
                _, log_prob = self.agent.policy_networks[group_idx].get_action(state_tensor)
                log_prob = log_prob.item()
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # Store data
            episode_data['states'].append(state)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['values'].append(value)
            episode_data['log_probs'].append(log_prob)
            episode_data['group_indices'].append(group_idx)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if terminated or truncated:
                episode_data['done'] = True
                break
        
        # Add final value for bootstrap
        if not episode_data['done']:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                final_value = self.agent.value_networks[group_idx](state_tensor).item()
            episode_data['values'].append(final_value)
        else:
            episode_data['values'].append(0.0)
        
        episode_data['episode_reward'] = episode_reward
        episode_data['episode_length'] = episode_length
        
        return episode_data
    
    def update_policies(self, episode_data: Dict):
        """
        Update policies using GRPO algorithm.
        
        Args:
            episode_data: Data collected from episode
        """
        group_idx = episode_data['group_indices'][0]  # Use first group for this episode
        
        # Compute advantages and returns
        advantages, returns = self.agent.compute_gae_advantages(
            episode_data['rewards'],
            episode_data['values'][:-1],  # Exclude final bootstrap value
            episode_data['values'][1:]     # Use next values
        )
        
        # Update policy
        self.agent.update_policy(
            group_idx,
            episode_data['states'],
            episode_data['actions'],
            advantages,
            returns,
            episode_data['log_probs']
        )
        
        # Update group performance
        self.agent.update_group_performance(group_idx, episode_data['episode_reward'])
    
    def train(self):
        """
        Main training loop for GRPO algorithm.
        """
        logger.info("Starting GRPO training...")
        
        # Training loop
        for episode in tqdm(range(self.config.num_episodes), desc="Training GRPO"):
            # Collect episode data
            episode_data = self.collect_episode_data()
            
            # Update policies
            self.update_policies(episode_data)
            
            # Store statistics
            self.episode_rewards.append(episode_data['episode_reward'])
            self.episode_lengths.append(episode_data['episode_length'])
            self.group_performances_history.append(self.agent.group_performances.copy())
            
            # Logging
            if episode % self.config.log_interval == 0:
                self.log_training_progress(episode)
            
            # Save models
            if episode % self.config.save_interval == 0:
                self.save_models(episode)
            
            # Visualization
            if episode % (self.config.log_interval * 2) == 0:
                self.create_visualizations(episode)
        
        logger.info("Training completed!")
        self.finalize_training()
    
    def log_training_progress(self, episode: int):
        """Log training progress"""
        recent_rewards = self.episode_rewards[-self.config.log_interval:]
        recent_lengths = self.episode_lengths[-self.config.log_interval:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        std_reward = np.std(recent_rewards)
        
        logger.info(f"Episode {episode}: "
                   f"Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}, "
                   f"Avg Length: {avg_length:.2f}")
        
        # Log group performances
        logger.info(f"Group Performances: {self.agent.group_performances}")
        
        # Log uncertainty stats
        if hasattr(self.env, 'get_uncertainty_stats'):
            uncertainty_stats = self.env.get_uncertainty_stats()
            logger.info(f"Uncertainty Stats: {uncertainty_stats}")
    
    def create_visualizations(self, episode: int):
        """Create and save training visualizations"""
        # Episode rewards
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Moving average
        window_size = min(50, len(self.episode_rewards) // 10)
        if window_size > 1:
            moving_avg = pd.Series(self.episode_rewards).rolling(window=window_size).mean()
            plt.plot(moving_avg, label=f'Moving Avg ({window_size})', color='red')
            plt.legend()
        
        # Episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.grid(True)
        
        # Group performances
        plt.subplot(2, 2, 3)
        group_perfs = np.array(self.group_performances_history)
        for i in range(self.config.num_groups):
            plt.plot(group_perfs[:, i], label=f'Group {i}')
        plt.title('Group Performances Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True)
        
        # Group performance heatmap
        plt.subplot(2, 2, 4)
        if len(self.group_performances_history) > 10:
            recent_perfs = group_perfs[-50:]  # Last 50 episodes
            sns.heatmap(recent_perfs.T, cmap='viridis', cbar=True)
            plt.title('Group Performance Heatmap (Last 50 Episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Group')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'training_progress_episode_{episode}.png'))
        plt.close()
        
        # Training losses
        if self.agent.training_losses:
            plt.figure(figsize=(10, 6))
            
            losses = self.agent.training_losses
            policy_losses = [l['policy_loss'] for l in losses]
            value_losses = [l['value_loss'] for l in losses]
            entropy_losses = [l['entropy_loss'] for l in losses]
            
            plt.subplot(1, 3, 1)
            plt.plot(policy_losses)
            plt.title('Policy Loss')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(value_losses)
            plt.title('Value Loss')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot(entropy_losses)
            plt.title('Entropy Loss')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'training_losses_episode_{episode}.png'))
            plt.close()
    
    def save_models(self, episode: int):
        """Save model checkpoints"""
        model_path = os.path.join(self.models_dir, f'model_episode_{episode}.pth')
        self.agent.save_models(model_path)
        
        # Save training statistics
        stats = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'group_performances': self.group_performances_history,
            'config': self.config.__dict__
        }
        
        stats_path = os.path.join(self.run_dir, f'training_stats_episode_{episode}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def finalize_training(self):
        """Finalize training and create summary visualizations"""
        logger.info("Finalizing training...")
        
        # Create final summary plot
        plt.figure(figsize=(15, 10))
        
        # Episode rewards with trend
        plt.subplot(2, 3, 1)
        plt.plot(self.episode_rewards, alpha=0.3, color='blue')
        
        # Add trend line
        if len(self.episode_rewards) > 10:
            x = np.arange(len(self.episode_rewards))
            z = np.polyfit(x, self.episode_rewards, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
        
        plt.title('Episode Rewards (with trend)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Episode lengths
        plt.subplot(2, 3, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.grid(True)
        
        # Group performance evolution
        plt.subplot(2, 3, 3)
        group_perfs = np.array(self.group_performances_history)
        for i in range(self.config.num_groups):
            plt.plot(group_perfs[:, i], label=f'Group {i}', linewidth=2)
        plt.title('Group Performance Evolution')
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        plt.legend()
        plt.grid(True)
        
        # Reward distribution
        plt.subplot(2, 3, 4)
        plt.hist(self.episode_rewards, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Length distribution
        plt.subplot(2, 3, 5)
        plt.hist(self.episode_lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Final group performance comparison
        plt.subplot(2, 3, 6)
        final_performances = self.agent.group_performances
        bars = plt.bar(range(len(final_performances)), final_performances)
        plt.title('Final Group Performances')
        plt.xlabel('Group')
        plt.ylabel('Performance')
        plt.xticks(range(len(final_performances)))
        
        # Add value labels on bars
        for bar, value in zip(bars, final_performances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'final_training_summary.png'), dpi=300)
        plt.close()
        
        # Save final model
        final_model_path = os.path.join(self.models_dir, 'final_model.pth')
        self.agent.save_models(final_model_path)
        
        # Create training summary
        self.create_training_summary()
        
        logger.info(f"Training finalized. Results saved to {self.run_dir}")
    
    def create_training_summary(self):
        """Create a comprehensive training summary"""
        summary = {
            'training_config': self.config.__dict__,
            'final_statistics': {
                'total_episodes': len(self.episode_rewards),
                'final_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
                'final_avg_length': np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths),
                'max_reward': np.max(self.episode_rewards),
                'min_reward': np.min(self.episode_rewards),
                'reward_std': np.std(self.episode_rewards),
                'final_group_performances': self.agent.group_performances.tolist(),
                'best_group': int(np.argmax(self.agent.group_performances))
            },
            'training_curves': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'group_performances_history': self.group_performances_history
            }
        }
        
        summary_path = os.path.join(self.run_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Training summary created")


def run_training(config: GRPOConfig = None):
    """
    Main function to run GRPO training.
    
    Args:
        config: GRPO configuration (uses default if None)
    """
    if config is None:
        config = GRPOConfig()
    
    # Create trainer
    trainer = GRPOTrainer(config)
    
    # Run training
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Example usage
    config = GRPOConfig(
        num_episodes=500,
        group_size=4,
        num_groups=3,
        learning_rate=3e-4
    )
    
    trainer = run_training(config)
