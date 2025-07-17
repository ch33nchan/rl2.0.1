# Enhanced RL2 Features Documentation

## Overview

This document describes the enhanced features added to RL2 for improved reinforcement learning capabilities, including adaptive KL penalty mechanisms, multi-objective optimization, alternative advantage estimation methods, automated hyperparameter tuning, advanced memory optimization, and comprehensive experiment tracking.

## Table of Contents

1. [Adaptive KL Penalty Mechanisms](#adaptive-kl-penalty-mechanisms)
2. [Multi-Objective Optimization](#multi-objective-optimization)
3. [Alternative Advantage Estimation Methods](#alternative-advantage-estimation-methods)
4. [Automated Hyperparameter Tuning](#automated-hyperparameter-tuning)
5. [Advanced Memory Optimization](#advanced-memory-optimization)
6. [Adaptive Batch Sizing](#adaptive-batch-sizing)
7. [Experiment Tracking and Model Versioning](#experiment-tracking-and-model-versioning)
8. [Configuration Reference](#configuration-reference)
9. [Usage Examples](#usage-examples)

## Adaptive KL Penalty Mechanisms

### Overview
The adaptive KL penalty system automatically adjusts the KL divergence coefficient during training to maintain a target KL divergence, improving training stability and convergence.

### Supported Controllers

#### 1. Exponential Controller
- **Type**: `exponential`
- **Behavior**: Increases coefficient when KL > target, decreases when KL < target
- **Parameters**:
  - `exp_factor`: Multiplication factor for increases (default: 1.03)
  - `exp_decay`: Multiplication factor for decreases (default: 0.99)

#### 2. Linear Controller
- **Type**: `linear`
- **Behavior**: Linear adjustment based on KL ratio
- **Parameters**:
  - `linear_factor`: Adjustment strength (default: 0.5)

#### 3. PID Controller
- **Type**: `pid`
- **Behavior**: Proportional-Integral-Derivative control
- **Parameters**:
  - `kp`: Proportional gain (default: 0.1)
  - `ki`: Integral gain (default: 0.01)
  - `kd`: Derivative gain (default: 0.001)

#### 4. Schedule Controller
- **Type**: `schedule`
- **Behavior**: Follows predefined schedule
- **Parameters**:
  - `schedule_type`: Schedule type (`cosine`, `linear`, `exponential`, `warmup_cosine`)
  - `schedule_steps`: Number of steps for schedule (default: 1000)

### Configuration Example
```yaml
actor:
  kl:
    adaptive_kl:
      enabled: true
      controller_type: "exponential"
      target_kl: 0.01
      initial_coef: 0.2
      exp_factor: 1.03
      exp_decay: 0.99
```

### Usage in Code
```python
from RL2.algs import AdaptiveKLController

controller = AdaptiveKLController(
    initial_coef=0.2,
    target_kl=0.01,
    controller_type="exponential"
)

# During training
kl_coef = controller.update(current_kl_div, step)
```

## Multi-Objective Optimization

### Overview
The multi-objective optimization framework allows training with multiple competing objectives (reward, KL penalty, entropy) using Pareto frontier analysis.

### Supported Methods

#### 1. Weighted Sum
- **Method**: `weighted_sum`
- **Behavior**: Linear combination of objectives
- **Best for**: Simple multi-objective problems

#### 2. Tchebycheff Scalarization
- **Method**: `tchebycheff`
- **Behavior**: Minimizes maximum weighted deviation from ideal point
- **Best for**: Better Pareto frontier coverage

#### 3. Pareto Dominance
- **Method**: `pareto_dominance`
- **Behavior**: Ranking based on dominance relationships
- **Best for**: True multi-objective optimization

#### 4. Hypervolume
- **Method**: `hypervolume`
- **Behavior**: Maximizes hypervolume contribution
- **Best for**: Diversity-preserving optimization

### Configuration Example
```yaml
multi_objective:
  enabled: true
  objectives: ["reward", "kl_penalty", "entropy"]
  objective_weights:
    reward: 1.0
    kl_penalty: -0.1
    entropy: 0.01
  pareto_method: "weighted_sum"
  diversity_weight: 0.1
  archive_size: 100
```

### Usage in Code
```python
from RL2.algs import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(
    objectives=["reward", "kl_penalty", "entropy"],
    objective_weights={"reward": 1.0, "kl_penalty": -0.1, "entropy": 0.01},
    pareto_method="weighted_sum"
)

# During training
objectives_dict = {
    "reward": reward_value,
    "kl_penalty": -kl_div,
    "entropy": entropy_value
}

loss = optimizer.compute_pareto_loss(objectives_dict)
optimizer.update_pareto_archive(objectives_dict)
```

## Alternative Advantage Estimation Methods

### Overview
Beyond GAE and REINFORCE, RL2 now supports advanced advantage estimation methods for better sample efficiency and stability.

### Supported Methods

#### 1. V-trace
- **Method**: `vtrace`
- **Best for**: Off-policy learning, distributed training
- **Parameters**:
  - `rho_bar`: Clipping threshold for importance sampling (default: 1.0)
  - `c_bar`: Clipping threshold for trace cutting (default: 1.0)

#### 2. Retrace(λ)
- **Method**: `retrace`
- **Best for**: Safe off-policy corrections
- **Parameters**:
  - `lamda`: Trace decay parameter (default: 0.95)

#### 3. TD(λ)
- **Method**: `td_lambda`
- **Best for**: Bias-variance trade-off control
- **Parameters**:
  - `lamda`: Eligibility trace decay (default: 0.95)

#### 4. Clipped Importance Sampling
- **Method**: `clipped_is`
- **Best for**: Off-policy data with experience replay
- **Parameters**:
  - `clip_ratio`: Clipping ratio (default: 0.2)

#### 5. Multi-step Returns
- **Method**: `multistep`
- **Best for**: Long-term credit assignment
- **Parameters**:
  - `n_steps`: Number of steps (default: 5)
  - `bootstrap_method`: Bootstrapping strategy (`gae`, `td`, `mc`)

### Configuration Example
```yaml
adv:
  estimator: "vtrace"
  gamma: 0.99
  lamda: 0.95
  
  vtrace:
    rho_bar: 1.0
    c_bar: 1.0
  
  multistep:
    n_steps: 5
    bootstrap_method: "gae"
```

### Usage in Code
```python
from RL2.algs import compute_advantages_unified

compute_advantages_unified(
    data_list,
    method="vtrace",
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0
)
```

## Automated Hyperparameter Tuning

### Overview
The hyperparameter optimization system automatically searches for optimal hyperparameters using various optimization algorithms.

### Supported Optimizers

#### 1. Random Search
- **Type**: `random`
- **Best for**: Quick exploration, baseline comparison
- **Parameters**: None

#### 2. Grid Search
- **Type**: `grid`
- **Best for**: Systematic exploration, small parameter spaces
- **Parameters**:
  - `grid_size`: Number of points per dimension (default: 10)

#### 3. Bayesian Optimization
- **Type**: `bayesian`
- **Best for**: Sample-efficient optimization
- **Parameters**:
  - `exploration_weight`: Exploration vs exploitation balance (default: 0.1)

### Configuration Example
```yaml
hyperopt:
  enabled: true
  optimizer_type: "bayesian"
  n_trials: 100
  save_dir: "hyperopt_results"
  
  bayesian:
    exploration_weight: 0.1
```

### Usage in Code
```python
from RL2.hyperopt import HyperparameterTuner, create_ppo_hyperparameter_configs

# Create parameter configurations
param_configs = create_ppo_hyperparameter_configs()

# Create tuner
tuner = HyperparameterTuner(
    optimizer_type="bayesian",
    param_configs=param_configs,
    objective_name="reward",
    maximize=True,
    n_trials=100
)

# Define training function
def train_function(params):
    # Update config with params
    updated_config = update_config_with_params(config, params)
    
    # Train model
    trainer = PPOTrainer(updated_config)
    results = trainer.train()
    
    return results

# Run optimization
best_params = tuner.optimize(train_function)
```

## Advanced Memory Optimization

### Overview
The memory optimization system provides comprehensive memory management including gradient checkpointing, CPU offloading, and automatic garbage collection.

### Features

#### 1. Gradient Checkpointing
- **Feature**: `enable_gradient_checkpointing`
- **Benefit**: Reduces memory usage by recomputing gradients
- **Trade-off**: Slightly increased computation time

#### 2. Activation Checkpointing
- **Feature**: `enable_activation_checkpointing`
- **Benefit**: Saves memory by checkpointing transformer layers
- **Trade-off**: Recomputation overhead

#### 3. CPU Offloading
- **Feature**: `enable_cpu_offload`
- **Benefit**: Moves tensors to CPU when not needed
- **Trade-off**: CPU-GPU transfer overhead

#### 4. Memory Profiling
- **Feature**: `profiling.enabled`
- **Benefit**: Detailed memory usage analysis
- **Output**: JSON profile with memory timeline

### Configuration Example
```yaml
memory_optimization:
  enabled: true
  enable_gradient_checkpointing: true
  enable_activation_checkpointing: true
  enable_cpu_offload: true
  memory_threshold: 0.8
  gc_threshold: 0.9
  
  profiling:
    enabled: true
    profile_interval: 1.0
    save_path: "memory_profile.json"
```

### Usage in Code
```python
from RL2.memory_optimizer import MemoryOptimizer, MemoryProfiler, memory_efficient_context

# Create optimizer
optimizer = MemoryOptimizer(
    enable_gradient_checkpointing=True,
    enable_cpu_offload=True,
    memory_threshold=0.8
)

# Create profiler
profiler = MemoryProfiler(
    enabled=True,
    profile_interval=1.0,
    save_path="memory_profile.json"
)

# Use memory-efficient context
with memory_efficient_context(optimizer):
    with profiler.profile_context("training"):
        # Training code here
        pass
```

## Adaptive Batch Sizing

### Overview
The adaptive batch sizing system automatically adjusts batch size based on available GPU memory and training performance.

### Features

#### 1. Memory-Based Adaptation
- Increases batch size when memory usage is low
- Decreases batch size when OOM occurs or memory is high

#### 2. Performance-Based Adaptation
- Considers training time and throughput
- Balances memory efficiency with training speed

#### 3. Success Rate Monitoring
- Tracks successful vs failed training steps
- Adjusts batch size based on success rate

### Configuration Example
```yaml
memory_optimization:
  adaptive_batch_sizing:
    enabled: true
    initial_batch_size: 8
    min_batch_size: 1
    max_batch_size: 64
    adaptation_factor: 0.1
```

### Usage in Code
```python
from RL2.memory_optimizer import AdaptiveBatchSizer

sizer = AdaptiveBatchSizer(
    initial_batch_size=8,
    min_batch_size=1,
    max_batch_size=64,
    adaptation_factor=0.1
)

# During training
memory_usage = get_memory_usage()
training_time = get_training_time()
success = training_succeeded()

new_batch_size = sizer.update_batch_size(memory_usage, training_time, success)
```

## Experiment Tracking and Model Versioning

### Overview
Comprehensive experiment tracking with support for MLflow and Weights & Biases, plus automatic model versioning.

### Features

#### 1. Multi-Backend Support
- **MLflow**: Enterprise-grade experiment tracking
- **Weights & Biases**: Cloud-based experiment management
- **Unified API**: Single interface for both backends

#### 2. Automatic Logging
- **Metrics**: Loss, reward, KL divergence, entropy
- **Parameters**: All hyperparameters
- **Artifacts**: Models, configs, memory profiles
- **System Metrics**: CPU, GPU, memory usage

#### 3. Model Versioning
- **Automatic versioning**: Semantic versioning
- **Model registry**: Central model repository
- **Stage management**: Development, staging, production
- **Model comparison**: Compare different versions

### Configuration Example
```yaml
experiment_tracking:
  enabled: true
  experiment_name: "RL2_PPO_Experiment"
  enable_mlflow: true
  enable_wandb: true
  log_models: true
  log_artifacts: true
  log_system_metrics: true
  
  model_versioning:
    enabled: true
    enable_auto_versioning: true
```

### Usage in Code
```python
from RL2.experiment_tracking import ExperimentTracker, ExperimentConfig, ModelVersioning

# Create tracker
config = ExperimentConfig(
    experiment_name="RL2_PPO_Experiment",
    enable_mlflow=True,
    enable_wandb=True
)

tracker = ExperimentTracker(config)

# Log metrics
tracker.log_metrics({"reward": 0.5, "kl_div": 0.01}, step=100)

# Log model
tracker.log_model(model, "ppo_actor", step=100)

# Model versioning
versioning = ModelVersioning(tracker)
version = versioning.register_model(model, "ppo_actor", description="Best model")
```

## Configuration Reference

### Complete Configuration Schema

```yaml
# Data configuration
data:
  train_data_path: null
  test_data_path: null
  prompts_per_rollout: null
  responses_per_prompt: null

# Actor configuration
actor:
  model_name: null
  lr: 1e-6
  clip: 0.2
  
  kl:
    coef: 0.0
    adaptive_kl:
      enabled: false
      controller_type: "exponential"
      target_kl: 0.01
      initial_coef: 0.2
      exp_factor: 1.03
      exp_decay: 0.99
      min_coef: 0.001
      max_coef: 10.0
    
    constraint_optimization:
      enabled: false
      kl_constraint: 0.01
      entropy_constraint: 0.01
      use_lagrangian: true
      lagrangian_lr: 0.01

# Advantage estimation
adv:
  estimator: "gae"  # gae, reinforce, vtrace, retrace, td_lambda, clipped_is, multistep
  gamma: 0.99
  lamda: 0.95
  
  vtrace:
    rho_bar: 1.0
    c_bar: 1.0
  
  retrace:
    lamda: 0.95
  
  clipped_is:
    clip_ratio: 0.2
  
  multistep:
    n_steps: 5
    bootstrap_method: "gae"

# Multi-objective optimization
multi_objective:
  enabled: false
  objectives: ["reward", "kl_penalty", "entropy"]
  objective_weights:
    reward: 1.0
    kl_penalty: -0.1
    entropy: 0.01
  pareto_method: "weighted_sum"
  diversity_weight: 0.1
  archive_size: 100

# Hyperparameter optimization
hyperopt:
  enabled: false
  optimizer_type: "random"
  n_trials: 100
  save_dir: "hyperopt_results"
  
  bayesian:
    exploration_weight: 0.1
  
  grid:
    grid_size: 10

# Memory optimization
memory_optimization:
  enabled: true
  enable_gradient_checkpointing: true
  enable_mixed_precision: true
  enable_cpu_offload: true
  enable_activation_checkpointing: true
  memory_threshold: 0.8
  gc_threshold: 0.9
  
  profiling:
    enabled: false
    profile_interval: 1.0
    save_path: "memory_profile.json"
  
  adaptive_batch_sizing:
    enabled: false
    initial_batch_size: 8
    min_batch_size: 1
    max_batch_size: 64
    adaptation_factor: 0.1

# Experiment tracking
experiment_tracking:
  enabled: true
  experiment_name: "RL2_PPO_Experiment"
  enable_mlflow: true
  enable_wandb: true
  log_models: true
  log_artifacts: true
  log_system_metrics: true
  
  model_versioning:
    enabled: false
    enable_auto_versioning: true
```

## Usage Examples

### Basic Usage with Enhanced Features

```python
import hydra
from omegaconf import OmegaConf
from RL2.trainer.ppo import PPOTrainer  # Enhanced trainer

@hydra.main(config_path="config", config_name="ppo")
def main(config):
    # Enable adaptive KL penalty
    config.actor.kl.adaptive_kl.enabled = True
    config.actor.kl.adaptive_kl.controller_type = "exponential"
    
    # Enable memory optimization
    config.memory_optimization.enabled = True
    config.memory_optimization.profiling.enabled = True
    
    # Enable experiment tracking
    config.experiment_tracking.enabled = True
    config.experiment_tracking.experiment_name = "my_experiment"
    
    # Create and run trainer
    trainer = PPOTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
```

### Hyperparameter Optimization

```python
from RL2.hyperopt import HyperparameterTuner, create_ppo_hyperparameter_configs

# Create parameter configurations
param_configs = create_ppo_hyperparameter_configs()

# Add custom parameters
param_configs.append(
    HyperparameterConfig(
        name="custom_param",
        type="float",
        low=0.1,
        high=1.0,
        default=0.5
    )
)

# Create tuner
tuner = HyperparameterTuner(
    optimizer_type="bayesian",
    param_configs=param_configs,
    n_trials=50
)

# Run optimization
best_params = tuner.optimize(training_function)
```

### Memory Optimization

```python
from RL2.memory_optimizer import MemoryOptimizer, MemoryProfiler

# Create optimizer and profiler
optimizer = MemoryOptimizer(memory_threshold=0.8)
profiler = MemoryProfiler(enabled=True)

# Optimize model
model = optimizer.optimize_model_memory(model)

# Find optimal batch size
optimal_batch_size = optimizer.optimize_batch_size(model, sample_input)

# Memory-efficient forward pass
output = optimizer.memory_efficient_forward(model, input_tensor)
```

### Advanced Multi-Objective Training

```python
from RL2.algs import MultiObjectiveOptimizer

# Create multi-objective optimizer
mo_optimizer = MultiObjectiveOptimizer(
    objectives=["reward", "kl_penalty", "entropy", "diversity"],
    objective_weights={
        "reward": 1.0,
        "kl_penalty": -0.1,
        "entropy": 0.05,
        "diversity": 0.02
    },
    pareto_method="tchebycheff"
)

# During training
objectives = {
    "reward": reward_value,
    "kl_penalty": -kl_div,
    "entropy": entropy_value,
    "diversity": diversity_metric
}

loss = mo_optimizer.compute_pareto_loss(objectives)
mo_optimizer.update_pareto_archive(objectives)

# Get Pareto front
pareto_front = mo_optimizer.get_pareto_front()
```

## Best Practices

### 1. Memory Optimization
- Enable gradient checkpointing for models > 7B parameters
- Use CPU offloading for models > 30B parameters
- Set memory threshold to 0.8 for stable training
- Enable memory profiling for optimization

### 2. Adaptive KL Penalty
- Use exponential controller for stable training
- Set target KL to 0.01 for most applications
- Use PID controller for fine-grained control
- Monitor KL history for trend analysis

### 3. Multi-Objective Optimization
- Start with weighted sum for simplicity
- Use Tchebycheff for better Pareto coverage
- Set diversity weight to 0.1-0.2
- Monitor Pareto archive size

### 4. Hyperparameter Tuning
- Start with random search for exploration
- Use Bayesian optimization for sample efficiency
- Define reasonable parameter bounds
- Run 50-100 trials for convergence

### 5. Experiment Tracking
- Always enable system metrics logging
- Use descriptive experiment names
- Tag experiments with relevant metadata
- Enable model versioning for production

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
- Enable gradient checkpointing
- Reduce batch size
- Enable CPU offloading
- Use mixed precision training

#### 2. KL Divergence Instability
- Enable adaptive KL penalty
- Use constraint optimization
- Reduce learning rate
- Check advantage estimation method

#### 3. Slow Training
- Enable memory optimization
- Use appropriate batch size
- Check system resource usage
- Consider model parallelization

#### 4. Poor Convergence
- Try different advantage estimation methods
- Enable multi-objective optimization
- Run hyperparameter tuning
- Check data quality and preprocessing

### Performance Optimization

#### 1. Memory
- Use gradient checkpointing for large models
- Enable CPU offloading for extreme cases
- Monitor memory usage with profiler
- Optimize batch size automatically

#### 2. Compute
- Use mixed precision training
- Enable Flash Attention if available
- Optimize sequence lengths
- Use efficient attention mechanisms

#### 3. I/O
- Preprocess data efficiently
- Use memory mapping for large datasets
- Implement efficient data loading
- Consider data caching strategies

## Contributing

To contribute new features:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Add configuration options
6. Include usage examples

## License

This enhanced version of RL2 is released under the Apache 2.0 License, same as the original RL2 project.
