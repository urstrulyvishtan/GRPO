"""
Evaluation Module for GRPO Algorithm

This module provides comprehensive evaluation capabilities for the GRPO algorithm,
including performance metrics, robustness testing, and comparison with baseline methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import json
import os
from datetime import datetime
import pandas as pd
from scipy import stats

from config import GRPOConfig
from grpo_agent import GRPOAgent
from uncertain_env import create_uncertain_env, UncertainCartPoleEnv, UncertaintyConfig
import gymnasium as gym

logger = logging.getLogger(__name__)

class GRPOEvaluator:
    """
    Comprehensive evaluator for GRPO algorithm.
    
    This class provides various evaluation metrics and testing capabilities:
    - Performance evaluation across different uncertainty levels
    - Robustness testing
    - Comparison with baseline methods
    - Statistical analysis of results
    """
    
    def __init__(self, config: GRPOConfig, agent: GRPOAgent):
        self.config = config
        self.agent = agent
        
        # Evaluation results storage
        self.evaluation_results = {}
        
        # Setup logging
        self.setup_logging()
        
        logger.info("Initialized GRPO evaluator")
    
    def setup_logging(self):
        """Setup evaluation logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = os.path.join(self.config.log_dir, f"evaluation_{timestamp}")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = os.path.join(self.eval_dir, "plots")
        self.results_dir = os.path.join(self.eval_dir, "results")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_performance(self, num_episodes: int = 100, 
                           uncertainty_levels: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate GRPO performance across different uncertainty levels.
        
        Args:
            num_episodes: Number of episodes to evaluate
            uncertainty_levels: List of uncertainty levels to test
            
        Returns:
            Dictionary containing evaluation results
        """
        if uncertainty_levels is None:
            uncertainty_levels = ['low', 'medium', 'high']
        
        logger.info(f"Evaluating GRPO performance across {len(uncertainty_levels)} uncertainty levels")
        
        results = {}
        
        for uncertainty_level in uncertainty_levels:
            logger.info(f"Evaluating uncertainty level: {uncertainty_level}")
            
            # Create environment with specific uncertainty level
            env = create_uncertain_env('cartpole', uncertainty_level)
            
            # Run evaluation episodes
            episode_rewards = []
            episode_lengths = []
            group_selections = []
            
            for episode in tqdm(range(num_episodes), desc=f"Evaluating {uncertainty_level}"):
                reward, length, group_selection = self._run_evaluation_episode(env)
                episode_rewards.append(reward)
                episode_lengths.append(length)
                group_selections.append(group_selection)
            
            # Store results
            results[uncertainty_level] = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'group_selections': group_selections,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths),
                'success_rate': np.mean([r > 195 for r in episode_rewards])  # CartPole success threshold
            }
        
        self.evaluation_results['performance'] = results
        return results
    
    def _run_evaluation_episode(self, env: gym.Env) -> Tuple[float, int, List[int]]:
        """
        Run a single evaluation episode.
        
        Args:
            env: Environment to evaluate on
            
        Returns:
            Tuple of (total_reward, episode_length, group_selections)
        """
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        group_selections = []
        
        for step in range(500):  # Max episode length
            action, group_idx = self.agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            episode_length += 1
            group_selections.append(group_idx)
            
            if terminated or truncated:
                break
        
        return total_reward, episode_length, group_selections
    
    def evaluate_robustness(self, num_episodes: int = 50) -> Dict[str, Any]:
        """
        Evaluate robustness of GRPO to various perturbations.
        
        Args:
            num_episodes: Number of episodes per perturbation test
            
        Returns:
            Dictionary containing robustness evaluation results
        """
        logger.info("Evaluating GRPO robustness to perturbations")
        
        # Define perturbation tests
        perturbations = {
            'observation_noise': {
                'observation_noise_std': 0.2,
                'action_noise_std': 0.0,
                'transition_noise_std': 0.0,
                'reward_noise_std': 0.0
            },
            'action_noise': {
                'observation_noise_std': 0.0,
                'action_noise_std': 0.1,
                'transition_noise_std': 0.0,
                'reward_noise_std': 0.0
            },
            'transition_noise': {
                'observation_noise_std': 0.0,
                'action_noise_std': 0.0,
                'transition_noise_std': 0.05,
                'reward_noise_std': 0.0
            },
            'reward_noise': {
                'observation_noise_std': 0.0,
                'action_noise_std': 0.0,
                'transition_noise_std': 0.0,
                'reward_noise_std': 0.05
            },
            'combined_perturbations': {
                'observation_noise_std': 0.1,
                'action_noise_std': 0.05,
                'transition_noise_std': 0.02,
                'reward_noise_std': 0.02
            }
        }
        
        results = {}
        
        for perturbation_name, perturbation_config in perturbations.items():
            logger.info(f"Testing robustness to: {perturbation_name}")
            
            # Create environment with perturbation
            uncertainty_config = UncertaintyConfig(**perturbation_config)
            env = UncertainCartPoleEnv(uncertainty_config)
            
            # Run evaluation episodes
            episode_rewards = []
            
            for episode in tqdm(range(num_episodes), desc=f"Testing {perturbation_name}"):
                reward, _, _ = self._run_evaluation_episode(env)
                episode_rewards.append(reward)
            
            # Store results
            results[perturbation_name] = {
                'episode_rewards': episode_rewards,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'success_rate': np.mean([r > 195 for r in episode_rewards])
            }
        
        self.evaluation_results['robustness'] = results
        return results
    
    def evaluate_group_diversity(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate diversity and specialization of different groups.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary containing group diversity analysis
        """
        logger.info("Evaluating group diversity and specialization")
        
        env = create_uncertain_env('cartpole', 'medium')
        
        # Test each group individually
        group_results = {}
        
        for group_idx in range(self.config.num_groups):
            logger.info(f"Evaluating group {group_idx}")
            
            episode_rewards = []
            episode_lengths = []
            
            for episode in tqdm(range(num_episodes), desc=f"Group {group_idx}"):
                reward, length, _ = self._run_evaluation_episode_with_group(env, group_idx)
                episode_rewards.append(reward)
                episode_lengths.append(length)
            
            group_results[f'group_{group_idx}'] = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths)
            }
        
        # Analyze group diversity
        group_means = [group_results[f'group_{i}']['mean_reward'] for i in range(self.config.num_groups)]
        group_stds = [group_results[f'group_{i}']['std_reward'] for i in range(self.config.num_groups)]
        
        diversity_metrics = {
            'group_means': group_means,
            'group_stds': group_stds,
            'mean_difference': np.std(group_means),
            'coefficient_of_variation': np.std(group_means) / np.mean(group_means) if np.mean(group_means) > 0 else 0,
            'group_performance_range': np.max(group_means) - np.min(group_means)
        }
        
        results = {
            'group_results': group_results,
            'diversity_metrics': diversity_metrics
        }
        
        self.evaluation_results['group_diversity'] = results
        return results
    
    def _run_evaluation_episode_with_group(self, env: gym.Env, group_idx: int) -> Tuple[float, int, List[int]]:
        """
        Run evaluation episode using a specific group.
        
        Args:
            env: Environment
            group_idx: Specific group to use
            
        Returns:
            Tuple of (total_reward, episode_length, group_selections)
        """
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        group_selections = []
        
        for step in range(500):
            action, _ = self.agent.get_action(state, group_idx)
            state, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            episode_length += 1
            group_selections.append(group_idx)
            
            if terminated or truncated:
                break
        
        return total_reward, episode_length, group_selections
    
    def compare_with_baselines(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Compare GRPO with baseline methods.
        
        Args:
            num_episodes: Number of episodes for comparison
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info("Comparing GRPO with baseline methods")
        
        # Create environments
        env_medium = create_uncertain_env('cartpole', 'medium')
        env_high = create_uncertain_env('cartpole', 'high')
        
        # Test GRPO
        grpo_results = {}
        
        for env_name, env in [('medium', env_medium), ('high', env_high)]:
            episode_rewards = []
            
            for episode in tqdm(range(num_episodes), desc=f"GRPO on {env_name}"):
                reward, _, _ = self._run_evaluation_episode(env)
                episode_rewards.append(reward)
            
            grpo_results[env_name] = {
                'episode_rewards': episode_rewards,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards)
            }
        
        # Test single policy (best group only)
        single_policy_results = {}
        
        best_group = np.argmax(self.agent.group_performances)
        
        for env_name, env in [('medium', env_medium), ('high', env_high)]:
            episode_rewards = []
            
            for episode in tqdm(range(num_episodes), desc=f"Single Policy on {env_name}"):
                reward, _, _ = self._run_evaluation_episode_with_group(env, best_group)
                episode_rewards.append(reward)
            
            single_policy_results[env_name] = {
                'episode_rewards': episode_rewards,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards)
            }
        
        # Statistical comparison
        comparison_results = {}
        
        for env_name in ['medium', 'high']:
            grpo_rewards = grpo_results[env_name]['episode_rewards']
            single_rewards = single_policy_results[env_name]['episode_rewards']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(grpo_rewards, single_rewards)
            
            comparison_results[env_name] = {
                'grpo_mean': np.mean(grpo_rewards),
                'single_policy_mean': np.mean(single_rewards),
                'improvement': np.mean(grpo_rewards) - np.mean(single_rewards),
                'improvement_percent': (np.mean(grpo_rewards) - np.mean(single_rewards)) / np.mean(single_rewards) * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        results = {
            'grpo_results': grpo_results,
            'single_policy_results': single_policy_results,
            'comparison': comparison_results
        }
        
        self.evaluation_results['baseline_comparison'] = results
        return results
    
    def create_evaluation_plots(self):
        """Create comprehensive evaluation visualizations"""
        logger.info("Creating evaluation visualizations")
        
        # Performance across uncertainty levels
        if 'performance' in self.evaluation_results:
            self._plot_performance_comparison()
        
        # Robustness analysis
        if 'robustness' in self.evaluation_results:
            self._plot_robustness_analysis()
        
        # Group diversity
        if 'group_diversity' in self.evaluation_results:
            self._plot_group_diversity()
        
        # Baseline comparison
        if 'baseline_comparison' in self.evaluation_results:
            self._plot_baseline_comparison()
    
    def _plot_performance_comparison(self):
        """Plot performance across uncertainty levels"""
        results = self.evaluation_results['performance']
        
        plt.figure(figsize=(12, 8))
        
        # Reward comparison
        plt.subplot(2, 2, 1)
        uncertainty_levels = list(results.keys())
        means = [results[level]['mean_reward'] for level in uncertainty_levels]
        stds = [results[level]['std_reward'] for level in uncertainty_levels]
        
        bars = plt.bar(uncertainty_levels, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Performance Across Uncertainty Levels')
        plt.ylabel('Mean Reward')
        plt.grid(True, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
        
        # Success rate comparison
        plt.subplot(2, 2, 2)
        success_rates = [results[level]['success_rate'] for level in uncertainty_levels]
        bars = plt.bar(uncertainty_levels, success_rates, alpha=0.7, color='green')
        plt.title('Success Rate Across Uncertainty Levels')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        plt.grid(True, axis='y')
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # Reward distributions
        plt.subplot(2, 2, 3)
        for level in uncertainty_levels:
            rewards = results[level]['episode_rewards']
            plt.hist(rewards, alpha=0.6, label=level, bins=20)
        plt.title('Reward Distributions')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # Episode length comparison
        plt.subplot(2, 2, 4)
        lengths = [results[level]['mean_length'] for level in uncertainty_levels]
        length_stds = [results[level]['std_length'] for level in uncertainty_levels]
        
        bars = plt.bar(uncertainty_levels, lengths, yerr=length_stds, capsize=5, alpha=0.7, color='orange')
        plt.title('Episode Length Across Uncertainty Levels')
        plt.ylabel('Mean Length')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'performance_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_robustness_analysis(self):
        """Plot robustness analysis"""
        results = self.evaluation_results['robustness']
        
        plt.figure(figsize=(12, 6))
        
        perturbations = list(results.keys())
        means = [results[p]['mean_reward'] for p in perturbations]
        stds = [results[p]['std_reward'] for p in perturbations]
        
        bars = plt.bar(perturbations, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Robustness to Different Perturbations')
        plt.ylabel('Mean Reward')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'robustness_analysis.png'), dpi=300)
        plt.close()
    
    def _plot_group_diversity(self):
        """Plot group diversity analysis"""
        results = self.evaluation_results['group_diversity']
        
        plt.figure(figsize=(12, 8))
        
        # Group performance comparison
        plt.subplot(2, 2, 1)
        group_results = results['group_results']
        group_names = list(group_results.keys())
        group_means = [group_results[name]['mean_reward'] for name in group_names]
        group_stds = [group_results[name]['std_reward'] for name in group_names]
        
        bars = plt.bar(group_names, group_means, yerr=group_stds, capsize=5, alpha=0.7)
        plt.title('Individual Group Performance')
        plt.ylabel('Mean Reward')
        plt.grid(True, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars, group_means, group_stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
        
        # Group diversity metrics
        plt.subplot(2, 2, 2)
        diversity_metrics = results['diversity_metrics']
        metrics = ['Mean Difference', 'Coefficient of Variation', 'Performance Range']
        values = [
            diversity_metrics['mean_difference'],
            diversity_metrics['coefficient_of_variation'],
            diversity_metrics['group_performance_range']
        ]
        
        bars = plt.bar(metrics, values, alpha=0.7, color='purple')
        plt.title('Group Diversity Metrics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        # Reward distributions by group
        plt.subplot(2, 2, 3)
        for name in group_names:
            rewards = group_results[name]['episode_rewards']
            plt.hist(rewards, alpha=0.6, label=name, bins=15)
        plt.title('Reward Distributions by Group')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # Group performance heatmap
        plt.subplot(2, 2, 4)
        group_data = []
        for name in group_names:
            group_data.append(group_results[name]['episode_rewards'])
        
        sns.heatmap(group_data, cmap='viridis', cbar=True, 
                   yticklabels=group_names, xticklabels=False)
        plt.title('Group Performance Heatmap')
        plt.ylabel('Group')
        plt.xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'group_diversity.png'), dpi=300)
        plt.close()
    
    def _plot_baseline_comparison(self):
        """Plot baseline comparison"""
        results = self.evaluation_results['baseline_comparison']
        
        plt.figure(figsize=(12, 6))
        
        env_names = ['medium', 'high']
        grpo_means = [results['grpo_results'][env]['mean_reward'] for env in env_names]
        single_means = [results['single_policy_results'][env]['mean_reward'] for env in env_names]
        
        x = np.arange(len(env_names))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, grpo_means, width, label='GRPO', alpha=0.7)
        bars2 = plt.bar(x + width/2, single_means, width, label='Single Policy', alpha=0.7)
        
        plt.title('GRPO vs Single Policy Comparison')
        plt.ylabel('Mean Reward')
        plt.xlabel('Uncertainty Level')
        plt.xticks(x, env_names)
        plt.legend()
        plt.grid(True, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + 5,
                        f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'baseline_comparison.png'), dpi=300)
        plt.close()
    
    def save_evaluation_results(self):
        """Save all evaluation results to files"""
        logger.info("Saving evaluation results")
        
        # Save raw results
        results_path = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Create summary report
        self._create_summary_report()
    
    def _create_summary_report(self):
        """Create a comprehensive summary report"""
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'summary': {}
        }
        
        # Performance summary
        if 'performance' in self.evaluation_results:
            perf_results = self.evaluation_results['performance']
            report['summary']['performance'] = {
                level: {
                    'mean_reward': results['mean_reward'],
                    'std_reward': results['std_reward'],
                    'success_rate': results['success_rate']
                }
                for level, results in perf_results.items()
            }
        
        # Robustness summary
        if 'robustness' in self.evaluation_results:
            robust_results = self.evaluation_results['robustness']
            report['summary']['robustness'] = {
                perturbation: {
                    'mean_reward': results['mean_reward'],
                    'std_reward': results['std_reward'],
                    'success_rate': results['success_rate']
                }
                for perturbation, results in robust_results.items()
            }
        
        # Group diversity summary
        if 'group_diversity' in self.evaluation_results:
            div_results = self.evaluation_results['group_diversity']
            report['summary']['group_diversity'] = div_results['diversity_metrics']
        
        # Baseline comparison summary
        if 'baseline_comparison' in self.evaluation_results:
            comp_results = self.evaluation_results['baseline_comparison']
            report['summary']['baseline_comparison'] = comp_results['comparison']
        
        # Save report
        report_path = os.path.join(self.results_dir, 'evaluation_summary.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation summary saved to {report_path}")


def run_comprehensive_evaluation(config: GRPOConfig, agent: GRPOAgent) -> GRPOEvaluator:
    """
    Run comprehensive evaluation of GRPO algorithm.
    
    Args:
        config: GRPO configuration
        agent: Trained GRPO agent
        
    Returns:
        GRPOEvaluator with results
    """
    evaluator = GRPOEvaluator(config, agent)
    
    # Run all evaluations
    evaluator.evaluate_performance()
    evaluator.evaluate_robustness()
    evaluator.evaluate_group_diversity()
    evaluator.compare_with_baselines()
    
    # Create visualizations
    evaluator.create_evaluation_plots()
    
    # Save results
    evaluator.save_evaluation_results()
    
    return evaluator


if __name__ == "__main__":
    # Example usage
    from trainer import run_training
    
    # Train agent first
    config = GRPOConfig(num_episodes=200)
    trainer = run_training(config)
    
    # Evaluate trained agent
    evaluator = run_comprehensive_evaluation(config, trainer.agent)
