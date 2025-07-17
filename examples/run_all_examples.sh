#!/bin/bash
# Example usage for RL2 enhanced component demos

set -e

# Adaptive KL Controller Example
python examples/example_adaptive_kl.py

echo "---"
# Multi-Objective Optimizer Example
python examples/example_multi_objective.py

echo "---"
# Memory Optimizer Example
python examples/example_memory_optimizer.py

echo "---"
# Experiment Tracker Example
python examples/example_experiment_tracker.py

echo "---"
# Full Enhanced RL2 Example (main demo)
python examples/enhanced_rl2_example.py --config examples/enhanced_ppo_config.yaml
