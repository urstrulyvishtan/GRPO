#!/usr/bin/env python3
"""
Example script demonstrating GRPO usage.

This script shows how to use GRPO for training and evaluation
with different configurations and uncertainty levels.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import GRPOConfig
from trainer import run_training
from evaluator import run_comprehensive_evaluation
from uncertain_env import create_uncertain_env, UncertaintyConfig

def example_basic_training():
    """Example 1: Basic GRPO training"""
    print("=" * 60)
    print("EXAMPLE 1: Basic GRPO Training")
    print("=" * 60)
    
    # Create configuration
    config = GRPOConfig(
        num_episodes=200,
        num_groups=3,
        group_size=6,
        learning_rate=3e-4,
        log_dir="example_logs"
    )
    
    print(f"Configuration:")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Groups: {config.num_groups}")
    print(f"  Group Size: {config.group_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    
    # Train the agent
    print("\nStarting training...")
    trainer = run_training(config)
    
    # Print results
    print(f"\nTraining Results:")
    print(f"  Final average reward: {trainer.episode_rewards[-10:]:.2f}")
    print(f"  Best episode reward: {max(trainer.episode_rewards):.2f}")
    print(f"  Final group performances: {trainer.agent.group_performances}")
    
    return trainer

def example_custom_uncertainty():
    """Example 2: Custom uncertainty configuration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Uncertainty Configuration")
    print("=" * 60)
    
    # Create custom uncertainty configuration
    uncertainty_config = UncertaintyConfig(
        observation_noise_std=0.2,
        action_noise_std=0.1,
        transition_noise_std=0.05,
        reward_noise_std=0.03,
        reward_delay_prob=0.2,
        partial_obs_prob=0.1,
        non_stationary_prob=0.05
    )
    
    print("Custom Uncertainty Configuration:")
    print(f"  Observation Noise: {uncertainty_config.observation_noise_std}")
    print(f"  Action Noise: {uncertainty_config.action_noise_std}")
    print(f"  Transition Noise: {uncertainty_config.transition_noise_std}")
    print(f"  Reward Noise: {uncertainty_config.reward_noise_std}")
    print(f"  Reward Delay Probability: {uncertainty_config.reward_delay_prob}")
    print(f"  Partial Observability: {uncertainty_config.partial_obs_prob}")
    print(f"  Non-stationary Probability: {uncertainty_config.non_stationary_prob}")
    
    # Create environment with custom uncertainty
    env = create_uncertain_env('cartpole', 'high')
    
    # Test environment
    print("\nTesting custom environment...")
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"  Test episode reward: {total_reward:.2f}")
    print(f"  Test episode length: {step + 1}")
    
    return env

def example_evaluation():
    """Example 3: Comprehensive evaluation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Comprehensive Evaluation")
    print("=" * 60)
    
    # First train a model
    config = GRPOConfig(
        num_episodes=100,
        num_groups=2,
        group_size=4,
        log_dir="example_eval_logs"
    )
    
    print("Training model for evaluation...")
    trainer = run_training(config)
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluator = run_comprehensive_evaluation(config, trainer.agent)
    
    # Print evaluation summary
    if 'performance' in evaluator.evaluation_results:
        print("\nPerformance Evaluation Results:")
        perf_results = evaluator.evaluation_results['performance']
        for level, results in perf_results.items():
            print(f"  {level} uncertainty:")
            print(f"    Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"    Success rate: {results['success_rate']:.2f}")
    
    if 'robustness' in evaluator.evaluation_results:
        print("\nRobustness Evaluation Results:")
        robust_results = evaluator.evaluation_results['robustness']
        for perturbation, results in robust_results.items():
            print(f"  {perturbation}:")
            print(f"    Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"    Success rate: {results['success_rate']:.2f}")
    
    return evaluator

def example_group_analysis():
    """Example 4: Group diversity analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Group Diversity Analysis")
    print("=" * 60)
    
    # Train model with multiple groups
    config = GRPOConfig(
        num_episodes=150,
        num_groups=4,
        group_size=6,
        log_dir="example_group_logs"
    )
    
    print("Training model with multiple groups...")
    trainer = run_training(config)
    
    # Analyze group performances
    print("\nGroup Performance Analysis:")
    group_performances = trainer.agent.group_performances
    
    for i, performance in enumerate(group_performances):
        print(f"  Group {i}: {performance:.3f}")
    
    # Calculate diversity metrics
    mean_performance = group_performances.mean()
    std_performance = group_performances.std()
    cv = std_performance / mean_performance if mean_performance > 0 else 0
    
    print(f"\nDiversity Metrics:")
    print(f"  Mean performance: {mean_performance:.3f}")
    print(f"  Standard deviation: {std_performance:.3f}")
    print(f"  Coefficient of variation: {cv:.3f}")
    print(f"  Performance range: {group_performances.max() - group_performances.min():.3f}")
    
    # Find best and worst groups
    best_group = group_performances.argmax()
    worst_group = group_performances.argmin()
    
    print(f"\nGroup Rankings:")
    print(f"  Best group: {best_group} (performance: {group_performances[best_group]:.3f})")
    print(f"  Worst group: {worst_group} (performance: {group_performances[worst_group]:.3f})")
    
    return trainer

def main():
    """Run all examples"""
    print("GRPO Examples")
    print("=" * 60)
    print("This script demonstrates various GRPO usage patterns.")
    print("Each example shows different aspects of the algorithm.")
    
    try:
        # Example 1: Basic training
        trainer1 = example_basic_training()
        
        # Example 2: Custom uncertainty
        env = example_custom_uncertainty()
        
        # Example 3: Evaluation
        evaluator = example_evaluation()
        
        # Example 4: Group analysis
        trainer4 = example_group_analysis()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - example_logs/: Basic training logs and plots")
        print("  - example_eval_logs/: Evaluation logs and plots")
        print("  - example_group_logs/: Group analysis logs and plots")
        print("\nCheck the generated plots and logs for detailed results.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed correctly.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
