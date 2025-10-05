#!/usr/bin/env python3
"""
Test script to verify GRPO implementation.

This script runs basic tests to ensure all components work correctly.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import gymnasium as gym
        print("✓ Core dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core dependencies: {e}")
        return False
    
    try:
        from config import GRPOConfig
        from grpo_agent import GRPOAgent
        from uncertain_env import create_uncertain_env
        from trainer import GRPOTrainer
        from evaluator import GRPOEvaluator
        print("✓ GRPO modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GRPO modules: {e}")
        return False
    
    return True

def test_config():
    """Test configuration creation"""
    print("Testing configuration...")
    
    try:
        from config import GRPOConfig
        
        config = GRPOConfig(
            num_episodes=10,
            num_groups=2,
            group_size=4
        )
        
        assert config.num_episodes == 10
        assert config.num_groups == 2
        assert config.group_size == 4
        print("✓ Configuration created successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_environment():
    """Test environment creation"""
    print("Testing environment...")
    
    try:
        from uncertain_env import create_uncertain_env
        
        env = create_uncertain_env('cartpole', 'medium')
        
        # Test reset
        state, _ = env.reset()
        assert len(state) == 4  # CartPole state dimension
        
        # Test step
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        assert len(next_state) == 4
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        print("✓ Environment created and tested successfully")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_agent():
    """Test GRPO agent creation"""
    print("Testing GRPO agent...")
    
    try:
        from config import GRPOConfig
        from grpo_agent import GRPOAgent
        
        config = GRPOConfig(num_groups=2, group_size=4)
        agent = GRPOAgent(config, state_dim=4, action_dim=2)
        
        # Test group selection
        group_idx = agent.select_group()
        assert 0 <= group_idx < config.num_groups
        
        # Test action selection
        state = [0.1, 0.2, 0.3, 0.4]
        action, selected_group = agent.get_action(state)
        
        assert 0 <= action < 2  # CartPole action space
        assert 0 <= selected_group < config.num_groups
        
        print("✓ GRPO agent created and tested successfully")
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

def test_training_step():
    """Test a single training step"""
    print("Testing training step...")
    
    try:
        from config import GRPOConfig
        from grpo_agent import GRPOAgent
        from uncertain_env import create_uncertain_env
        
        config = GRPOConfig(num_episodes=1, num_groups=2, group_size=4)
        agent = GRPOAgent(config, state_dim=4, action_dim=2)
        env = create_uncertain_env('cartpole', 'low')
        
        # Collect one episode
        state, _ = env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'group_indices': []
        }
        
        for step in range(10):  # Short episode
            action, group_idx = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Get value and log prob
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                value = agent.value_networks[group_idx](state_tensor).item()
                _, log_prob = agent.policy_networks[group_idx].get_action(state_tensor)
                log_prob = log_prob.item()
            
            episode_data['states'].append(state)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['values'].append(value)
            episode_data['log_probs'].append(log_prob)
            episode_data['group_indices'].append(group_idx)
            
            state = next_state
            if terminated or truncated:
                break
        
        # Add final value
        episode_data['values'].append(0.0)
        
        # Test policy update
        group_idx = episode_data['group_indices'][0]
        advantages, returns = agent.compute_gae_advantages(
            episode_data['rewards'],
            episode_data['values'][:-1],
            episode_data['values'][1:]
        )
        
        agent.update_policy(
            group_idx,
            episode_data['states'],
            episode_data['actions'],
            advantages,
            returns,
            episode_data['log_probs']
        )
        
        print("✓ Training step completed successfully")
        return True
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation functionality"""
    print("Testing evaluation...")
    
    try:
        from config import GRPOConfig
        from grpo_agent import GRPOAgent
        from evaluator import GRPOEvaluator
        
        config = GRPOConfig(num_groups=2, group_size=4)
        agent = GRPOAgent(config, state_dim=4, action_dim=2)
        evaluator = GRPOEvaluator(config, agent)
        
        # Test evaluation episode
        from uncertain_env import create_uncertain_env
        env = create_uncertain_env('cartpole', 'medium')
        
        reward, length, group_selections = evaluator._run_evaluation_episode(env)
        
        assert isinstance(reward, (int, float))
        assert isinstance(length, int)
        assert isinstance(group_selections, list)
        
        print("✓ Evaluation functionality tested successfully")
        return True
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("GRPO Implementation Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_environment,
        test_agent,
        test_training_step,
        test_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! GRPO implementation is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
