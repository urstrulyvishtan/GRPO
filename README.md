# GRPO: Group Relative Policy Optimization

A comprehensive implementation of Group Relative Policy Optimization (GRPO) algorithm with synthetic data and real-world uncertainties for reinforcement learning.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Algorithm Details](#algorithm-details)
- [Project Structure](#project-structure)
- [End-to-End Workflow](#end-to-end-workflow)
- [Examples](#examples)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

## Overview

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm that maintains multiple groups of policies and uses relative performance between groups to guide policy updates. This approach provides:

- **Robustness**: Multiple policy groups handle different types of uncertainties
- **Exploration**: Group diversity prevents premature convergence
- **Adaptability**: Dynamic group selection based on performance
- **Scalability**: Parallel training of multiple policy groups

This implementation includes:
- Complete GRPO algorithm with detailed explanations
- Synthetic RL environments with real-world uncertainties
- Comprehensive evaluation and visualization tools
- Easy-to-understand code with extensive documentation

## Features

### 🚀 Core Algorithm
- **Group-based Policy Optimization**: Multiple policy groups with relative performance tracking
- **Generalized Advantage Estimation (GAE)**: Advanced advantage computation
- **Clipped Policy Loss**: PPO-style policy updates with clipping
- **Value Function Learning**: Separate value network for each group

### 🌍 Real-World Uncertainties
- **Observation Noise**: Sensor measurement uncertainties
- **Action Execution Noise**: Imperfect action execution
- **Transition Model Uncertainty**: Model mismatch and dynamics noise
- **Reward Noise and Delays**: Delayed and noisy reward signals
- **Partial Observability**: Missing or masked observations
- **Non-stationary Dynamics**: Changing environment parameters

### 📊 Comprehensive Evaluation
- **Performance Analysis**: Across different uncertainty levels
- **Robustness Testing**: Against various perturbations
- **Group Diversity Analysis**: Specialization and diversity metrics
- **Baseline Comparisons**: Against single-policy methods
- **Statistical Analysis**: Significance testing and confidence intervals

### 📈 Visualization and Logging
- **Real-time Training Plots**: Episode rewards, lengths, and losses
- **Group Performance Tracking**: Evolution of group performances
- **Uncertainty Statistics**: Applied uncertainties and their effects
- **Comprehensive Reports**: JSON summaries and analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd GRPO
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Or install manually**:
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation
```bash
python -c "import torch, numpy, matplotlib; print('Installation successful!')"
```

## Quick Start

### 1. Run a Quick Demo
```bash
python main.py demo --episodes 50
```

### 2. Train a Model
```bash
python main.py train --episodes 500 --groups 4
```

### 3. Run Full Pipeline
```bash
python main.py pipeline --train-episodes 500 --eval-episodes 100
```

## Detailed Usage

### Command Line Interface

The project provides a comprehensive CLI with multiple commands:

#### Training
```bash
python main.py train [OPTIONS]

Options:
  --episodes INT           Number of training episodes (default: 1000)
  --groups INT             Number of policy groups (default: 4)
  --group-size INT         Number of policies per group (default: 8)
  --learning-rate FLOAT    Learning rate (default: 3e-4)
  --uncertainty-level STR  Uncertainty level: low/medium/high (default: medium)
  --log-dir STR            Directory for logging (default: logs)
```

#### Evaluation
```bash
python main.py evaluate --model-path PATH [OPTIONS]

Options:
  --model-path STR         Path to trained model (required)
  --episodes INT          Number of evaluation episodes (default: 100)
  --log-dir STR           Directory for logging (default: logs)
```

#### Demonstration
```bash
python main.py demo [OPTIONS]

Options:
  --episodes INT           Number of demo episodes (default: 50)
  --uncertainty-level STR  Uncertainty level: low/medium/high (default: medium)
```

#### Full Pipeline
```bash
python main.py pipeline [OPTIONS]

Options:
  --train-episodes INT    Number of training episodes (default: 500)
  --eval-episodes INT     Number of evaluation episodes (default: 100)
  --groups INT            Number of policy groups (default: 4)
  --log-dir STR           Directory for logging (default: logs)
```

### Programmatic Usage

#### Basic Training
```python
from config import GRPOConfig
from trainer import run_training

# Create configuration
config = GRPOConfig(
    num_episodes=1000,
    num_groups=4,
    group_size=8,
    learning_rate=3e-4
)

# Train the agent
trainer = run_training(config)
```

#### Evaluation
```python
from evaluator import run_comprehensive_evaluation

# Evaluate the trained agent
evaluator = run_comprehensive_evaluation(config, trainer.agent)
```

#### Custom Environment
```python
from uncertain_env import create_uncertain_env, UncertaintyConfig

# Create custom uncertainty configuration
uncertainty_config = UncertaintyConfig(
    observation_noise_std=0.1,
    action_noise_std=0.05,
    transition_noise_std=0.02,
    reward_noise_std=0.01
)

# Create environment
env = create_uncertain_env('cartpole', 'high')
```

## Algorithm Details

### GRPO Algorithm

GRPO extends traditional policy optimization by maintaining multiple groups of policies:

1. **Group Initialization**: Create N groups, each with M policies
2. **Group Selection**: Select group based on performance-weighted sampling
3. **Policy Execution**: Execute action from selected group's policy
4. **Performance Tracking**: Update group performance using episode rewards
5. **Policy Updates**: Update policies using GRPO loss with group-specific advantages
6. **Group Evolution**: Groups adapt based on relative performance

### Key Components

#### Policy Network
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        # Feedforward network with ReLU activations
        # Xavier weight initialization
        # Softmax output for action probabilities
```

#### Value Network
```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        # Separate value estimation network
        # Used for advantage computation
```

#### GRPO Loss
```python
# Policy loss with clipping
ratio = torch.exp(log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
policy_loss = -torch.min(surr1, surr2).mean()

# Value loss
value_loss = F.mse_loss(values, returns)

# Entropy loss for exploration
entropy_loss = -action_dist.entropy().mean()
```

### Uncertainty Types

#### 1. Observation Noise
- **Purpose**: Simulates sensor measurement errors
- **Implementation**: Gaussian noise added to state observations
- **Real-world example**: Camera blur, sensor drift

#### 2. Action Execution Noise
- **Purpose**: Simulates imperfect action execution
- **Implementation**: Random action flipping with small probability
- **Real-world example**: Motor control errors, actuator delays

#### 3. Transition Model Uncertainty
- **Purpose**: Simulates model mismatch and dynamics uncertainty
- **Implementation**: Noise added to force/control inputs
- **Real-world example**: Unmodeled dynamics, parameter drift

#### 4. Reward Noise and Delays
- **Purpose**: Simulates delayed and noisy reward signals
- **Implementation**: Random reward delays and Gaussian noise
- **Real-world example**: Delayed feedback, measurement noise

#### 5. Partial Observability
- **Purpose**: Simulates missing or masked observations
- **Implementation**: Random masking of state components
- **Real-world example**: Sensor failures, occlusions

#### 6. Non-stationary Dynamics
- **Purpose**: Simulates changing environment parameters
- **Implementation**: Random changes to gravity, mass, etc.
- **Real-world example**: Weather changes, wear and tear

## Project Structure

```
GRPO/
├── main.py                 # Main CLI script
├── setup.py               # Setup and installation script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── config.py              # Configuration classes
├── grpo_agent.py          # GRPO algorithm implementation
├── uncertain_env.py       # Synthetic environments with uncertainties
├── trainer.py             # Training loop and visualization
├── evaluator.py           # Evaluation and analysis tools
├── logs/                  # Training and evaluation logs
├── models/                # Saved model checkpoints
├── plots/                 # Generated plots and visualizations
├── results/               # Evaluation results and reports
└── data/                  # Data storage (if needed)
```

### File Descriptions

- **`config.py`**: Configuration classes for GRPO and uncertainty parameters
- **`grpo_agent.py`**: Core GRPO algorithm with policy/value networks
- **`uncertain_env.py`**: Synthetic environments with various uncertainty types
- **`trainer.py`**: Training pipeline with logging and visualization
- **`evaluator.py`**: Comprehensive evaluation tools and metrics
- **`main.py`**: Command-line interface for all functionality

## End-to-End Workflow

### 1. Environment Setup
```bash
# Install dependencies
python setup.py

# Verify installation
python -c "import torch, numpy, matplotlib; print('Ready!')"
```

### 2. Quick Demonstration
```bash
# Run a quick demo to see GRPO in action
python main.py demo --episodes 50 --uncertainty-level medium
```

### 3. Training Phase
```bash
# Train GRPO agent
python main.py train \
    --episodes 1000 \
    --groups 4 \
    --group-size 8 \
    --learning-rate 3e-4 \
    --uncertainty-level medium
```

**What happens during training:**
- Multiple policy groups are initialized
- Agent interacts with uncertain environment
- Policies are updated using GRPO algorithm
- Group performances are tracked and updated
- Training progress is logged and visualized

### 4. Evaluation Phase
```bash
# Evaluate trained model
python main.py evaluate \
    --model-path logs/grpo_run_*/models/final_model.pth \
    --episodes 100
```

**What happens during evaluation:**
- Performance across different uncertainty levels
- Robustness testing against perturbations
- Group diversity analysis
- Comparison with baseline methods
- Statistical significance testing

### 5. Full Pipeline
```bash
# Run complete training and evaluation pipeline
python main.py pipeline \
    --train-episodes 500 \
    --eval-episodes 100 \
    --groups 4
```

### 6. Results Analysis
After training and evaluation, you'll find:

- **Training logs**: `logs/grpo_run_*/training.log`
- **Model checkpoints**: `logs/grpo_run_*/models/`
- **Training plots**: `logs/grpo_run_*/plots/`
- **Evaluation results**: `logs/evaluation_*/results/`
- **Summary reports**: `logs/evaluation_*/evaluation_summary.json`

## Examples

### Example 1: Basic Training
```python
from config import GRPOConfig
from trainer import run_training

# Configure GRPO
config = GRPOConfig(
    num_episodes=500,
    num_groups=3,
    group_size=6,
    learning_rate=3e-4,
    uncertainty_level='medium'
)

# Train agent
trainer = run_training(config)

# Check results
print(f"Final group performances: {trainer.agent.group_performances}")
print(f"Best episode reward: {max(trainer.episode_rewards)}")
```

### Example 2: Custom Uncertainty Environment
```python
from uncertain_env import UncertainCartPoleEnv, UncertaintyConfig

# Create custom uncertainty configuration
uncertainty_config = UncertaintyConfig(
    observation_noise_std=0.15,
    action_noise_std=0.08,
    transition_noise_std=0.03,
    reward_noise_std=0.02,
    reward_delay_prob=0.15,
    partial_obs_prob=0.08,
    non_stationary_prob=0.03
)

# Create environment
env = UncertainCartPoleEnv(uncertainty_config)

# Test environment
state, _ = env.reset()
for step in range(100):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
```

### Example 3: Evaluation Analysis
```python
from evaluator import GRPOEvaluator

# Create evaluator
evaluator = GRPOEvaluator(config, trained_agent)

# Run comprehensive evaluation
performance_results = evaluator.evaluate_performance(num_episodes=100)
robustness_results = evaluator.evaluate_robustness(num_episodes=50)
diversity_results = evaluator.evaluate_group_diversity(num_episodes=100)

# Create visualizations
evaluator.create_evaluation_plots()

# Save results
evaluator.save_evaluation_results()
```

### Example 4: Multi-Task Environment
```python
from uncertain_env import MultiTaskUncertainEnv

# Create multi-task environment
env = MultiTaskUncertainEnv()

# Train on multiple tasks
for episode in range(1000):
    state, _ = env.reset()
    current_task = env.get_current_task()
    
    # Train on current task
    # ... training code ...
    
    if episode % 100 == 0:
        print(f"Current task: {current_task}")
```

## Results and Visualizations

### Training Visualizations

The training process generates several types of plots:

1. **Episode Rewards**: Shows learning progress over time
2. **Episode Lengths**: Tracks episode duration
3. **Group Performances**: Evolution of group performance over time
4. **Training Losses**: Policy, value, and entropy losses
5. **Group Performance Heatmap**: Visual representation of group diversity

### Evaluation Visualizations

The evaluation process creates comprehensive analysis plots:

1. **Performance Comparison**: Across different uncertainty levels
2. **Robustness Analysis**: Performance under various perturbations
3. **Group Diversity**: Individual group performance and diversity metrics
4. **Baseline Comparison**: GRPO vs single-policy methods
5. **Statistical Analysis**: Confidence intervals and significance tests

### Sample Results

Typical results from GRPO training:

- **Learning Curve**: Steady improvement in episode rewards
- **Group Specialization**: Different groups excel in different scenarios
- **Robustness**: Consistent performance across uncertainty levels
- **Diversity**: Groups maintain distinct behaviors and strategies

### Logging and Monitoring

All training and evaluation processes are logged with:

- **Real-time Progress**: Episode rewards, lengths, and group performances
- **Uncertainty Statistics**: Applied uncertainties and their effects
- **Training Metrics**: Losses, learning rates, and convergence
- **Evaluation Metrics**: Performance, robustness, and diversity measures

## Contributing

We welcome contributions to improve GRPO! Here's how you can help:

### Areas for Contribution

1. **Algorithm Improvements**:
   - New group selection strategies
   - Advanced uncertainty handling
   - Multi-objective optimization

2. **Environment Extensions**:
   - New uncertainty types
   - Additional RL environments
   - Real-world environment interfaces

3. **Evaluation Enhancements**:
   - New evaluation metrics
   - Additional baseline comparisons
   - Statistical analysis improvements

4. **Documentation**:
   - Code documentation
   - Tutorial notebooks
   - Algorithm explanations

### Development Setup

1. **Fork the repository**
2. **Create a development environment**:
   ```bash
   python -m venv grpo_dev
   source grpo_dev/bin/activate  # On Windows: grpo_dev\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Make your changes**
4. **Run tests**:
   ```bash
   python main.py demo --episodes 10
   python main.py train --episodes 50
   ```

5. **Submit a pull request**

### Code Style

- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Include docstrings for all functions and classes
- Write clear, descriptive variable names

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PPO Algorithm**: Base policy optimization approach
- **Gymnasium**: RL environment framework
- **PyTorch**: Deep learning framework
- **CartPole Environment**: Classic RL benchmark

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{grpo_implementation,
  title={GRPO: Group Relative Policy Optimization Implementation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/GRPO}
}
```

## Support

For questions, issues, or contributions:

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact [your-email@domain.com]

---

**Happy Learning with GRPO! 🚀**
