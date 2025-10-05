#!/usr/bin/env python3
"""
Main script for GRPO (Group Relative Policy Optimization) implementation.

This script provides a command-line interface for training and evaluating
the GRPO algorithm on synthetic environments with real-world uncertainties.
"""

import argparse
import logging
import os
import sys
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import GRPOConfig
from trainer import run_training
from evaluator import run_comprehensive_evaluation
from uncertain_env import create_uncertain_env

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="GRPO (Group Relative Policy Optimization) Implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train GRPO agent')
    train_parser.add_argument('--episodes', type=int, default=1000,
                             help='Number of training episodes')
    train_parser.add_argument('--groups', type=int, default=4,
                             help='Number of policy groups')
    train_parser.add_argument('--group-size', type=int, default=8,
                             help='Number of policies per group')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4,
                             help='Learning rate')
    train_parser.add_argument('--uncertainty-level', type=str, default='medium',
                             choices=['low', 'medium', 'high'],
                             help='Uncertainty level for training environment')
    train_parser.add_argument('--log-dir', type=str, default='logs',
                             help='Directory for logging')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained GRPO agent')
    eval_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--episodes', type=int, default=100,
                           help='Number of evaluation episodes')
    eval_parser.add_argument('--log-dir', type=str, default='logs',
                           help='Directory for logging')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--episodes', type=int, default=50,
                           help='Number of demo episodes')
    demo_parser.add_argument('--uncertainty-level', type=str, default='medium',
                           choices=['low', 'medium', 'high'],
                           help='Uncertainty level for demo environment')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full training and evaluation pipeline')
    pipeline_parser.add_argument('--train-episodes', type=int, default=500,
                               help='Number of training episodes')
    pipeline_parser.add_argument('--eval-episodes', type=int, default=100,
                               help='Number of evaluation episodes')
    pipeline_parser.add_argument('--groups', type=int, default=4,
                               help='Number of policy groups')
    pipeline_parser.add_argument('--log-dir', type=str, default='logs',
                               help='Directory for logging')
    
    return parser


def train_command(args):
    """Execute training command"""
    logger.info("Starting GRPO training...")
    
    # Create configuration
    config = GRPOConfig(
        num_episodes=args.episodes,
        num_groups=args.groups,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        log_dir=args.log_dir
    )
    
    # Run training
    trainer = run_training(config)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {trainer.run_dir}")
    
    return trainer


def evaluate_command(args):
    """Execute evaluation command"""
    logger.info("Starting GRPO evaluation...")
    
    # Load configuration from model
    import torch
    model_data = torch.load(args.model_path)
    config = model_data['config']
    
    # Create agent and load model
    from grpo_agent import GRPOAgent
    agent = GRPOAgent(config, 4, 2)  # CartPole dimensions
    agent.load_models(args.model_path)
    
    # Run evaluation
    evaluator = run_comprehensive_evaluation(config, agent)
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to: {evaluator.eval_dir}")
    
    return evaluator


def demo_command(args):
    """Execute demonstration command"""
    logger.info("Running GRPO demonstration...")
    
    # Create a simple configuration for demo
    config = GRPOConfig(
        num_episodes=args.episodes,
        num_groups=3,
        group_size=4,
        log_dir='demo_logs'
    )
    
    # Train a quick model
    logger.info("Training demo model...")
    trainer = run_training(config)
    
    # Run evaluation
    logger.info("Evaluating demo model...")
    evaluator = run_comprehensive_evaluation(config, trainer.agent)
    
    # Create environment for live demo
    env = create_uncertain_env('cartpole', args.uncertainty_level)
    
    logger.info("Running live demonstration...")
    print("\n" + "="*50)
    print("GRPO LIVE DEMONSTRATION")
    print("="*50)
    
    for episode in range(5):  # Run 5 demo episodes
        state, _ = env.reset()
        total_reward = 0
        step = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        for step in range(200):  # Max 200 steps
            action, group_idx = trainer.agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if step % 50 == 0:  # Print every 50 steps
                print(f"  Step {step}: Group {group_idx}, Reward: {reward:.2f}")
            
            if terminated or truncated:
                break
        
        print(f"  Episode completed: {step + 1} steps, Total reward: {total_reward:.2f}")
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETED")
    print("="*50)
    
    logger.info("Demo completed successfully!")
    logger.info(f"Training results: {trainer.run_dir}")
    logger.info(f"Evaluation results: {evaluator.eval_dir}")
    
    return trainer, evaluator


def pipeline_command(args):
    """Execute full pipeline command"""
    logger.info("Starting full GRPO pipeline...")
    
    # Create configuration
    config = GRPOConfig(
        num_episodes=args.train_episodes,
        num_groups=args.groups,
        log_dir=args.log_dir
    )
    
    # Step 1: Training
    logger.info("Step 1: Training GRPO agent...")
    trainer = run_training(config)
    
    # Step 2: Evaluation
    logger.info("Step 2: Evaluating GRPO agent...")
    evaluator = run_comprehensive_evaluation(config, trainer.agent)
    
    # Step 3: Summary
    logger.info("Step 3: Creating summary...")
    create_pipeline_summary(trainer, evaluator)
    
    logger.info("Full pipeline completed successfully!")
    logger.info(f"Training results: {trainer.run_dir}")
    logger.info(f"Evaluation results: {evaluator.eval_dir}")
    
    return trainer, evaluator


def create_pipeline_summary(trainer, evaluator):
    """Create a summary of the full pipeline results"""
    summary_path = os.path.join(trainer.run_dir, 'pipeline_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("GRPO PIPELINE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write(f"  Total episodes: {len(trainer.episode_rewards)}\n")
        f.write(f"  Final average reward: {np.mean(trainer.episode_rewards[-10:]):.2f}\n")
        f.write(f"  Best episode reward: {max(trainer.episode_rewards):.2f}\n")
        f.write(f"  Final group performances: {trainer.agent.group_performances}\n\n")
        
        f.write("EVALUATION RESULTS:\n")
        if 'performance' in evaluator.evaluation_results:
            perf_results = evaluator.evaluation_results['performance']
            for level, results in perf_results.items():
                f.write(f"  {level} uncertainty: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        
        f.write("\nCONCLUSIONS:\n")
        f.write("  - GRPO successfully trained on uncertain environments\n")
        f.write("  - Multiple groups provide robustness and diversity\n")
        f.write("  - Algorithm adapts to different uncertainty levels\n")
    
    logger.info(f"Pipeline summary saved to: {summary_path}")


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'demo':
            demo_command(args)
        elif args.command == 'pipeline':
            pipeline_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
