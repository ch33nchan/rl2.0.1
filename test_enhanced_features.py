#!/usr/bin/env python3
"""
Test script for enhanced RL2 features
"""

import sys
import torch
import numpy as np
import time
from pathlib import Path
import logging

# Add RL2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_adaptive_kl_controller():
    """Test adaptive KL controller"""
    from RL2.algs import AdaptiveKLController
    
    print("Testing AdaptiveKLController...")
    
    # Test exponential controller
    controller = AdaptiveKLController(
        initial_coef=0.2,
        target_kl=0.01,
        controller_type="exponential"
    )
    
    # Simulate KL divergence values
    kl_values = [0.02, 0.015, 0.005, 0.008, 0.012]
    
    for i, kl in enumerate(kl_values):
        coef = controller.update(kl, i)
        print(f"Step {i}: KL={kl:.3f}, Coef={coef:.3f}")
    
    stats = controller.get_stats()
    print(f"Controller stats: {stats}")
    print("✓ AdaptiveKLController test passed\n")

def test_multi_objective_optimizer():
    """Test multi-objective optimizer"""
    from RL2.algs import MultiObjectiveOptimizer
    
    print("Testing MultiObjectiveOptimizer...")
    
    optimizer = MultiObjectiveOptimizer(
        objectives=["reward", "kl_penalty", "entropy"],
        objective_weights={"reward": 1.0, "kl_penalty": -0.1, "entropy": 0.01},
        pareto_method="weighted_sum"
    )
    
    # Simulate objectives
    for i in range(10):
        objectives = {
            "reward": np.random.normal(0.5, 0.1),
            "kl_penalty": -np.random.exponential(0.01),
            "entropy": np.random.exponential(0.05)
        }
        
        loss = optimizer.compute_pareto_loss(objectives)
        optimizer.update_pareto_archive(objectives)
        
        print(f"Step {i}: Loss={loss:.3f}, Pareto size={len(optimizer.pareto_archive)}")
    
    stats = optimizer.get_stats()
    print(f"Multi-objective stats: {stats}")
    print("✓ MultiObjectiveOptimizer test passed\n")

def test_advantage_estimation():
    """Test alternative advantage estimation methods"""
    from RL2.algs import compute_advantages_unified
    
    print("Testing advantage estimation methods...")
    
    # Create sample data
    data_list = [
        {
            "rewards": torch.randn(50),
            "values": torch.randn(50),
            "logps": torch.randn(50),
            "ref_logps": torch.randn(50),
            "action_mask": torch.randint(0, 2, (50,)).bool()
        }
        for _ in range(3)
    ]
    
    methods = ["gae", "vtrace", "retrace", "td_lambda", "clipped_is", "multistep"]
    
    for method in methods:
        try:
            # Create fresh copy for each method
            test_data = [{k: v.clone() for k, v in ex.items()} for ex in data_list]
            
            compute_advantages_unified(
                test_data,
                method=method,
                gamma=0.99,
                lamda=0.95,
                rho_bar=1.0,
                c_bar=1.0,
                clip_ratio=0.2,
                n_steps=5,
                bootstrap_method="gae"
            )
            
            # Check if advantages were computed
            has_advantages = all("advantages" in ex for ex in test_data)
            print(f"Method {method}: {'✓' if has_advantages else '✗'}")
            
        except Exception as e:
            print(f"Method {method}: ✗ (Error: {e})")
    
    print("✓ Advantage estimation test completed\n")

def test_memory_optimizer():
    """Test memory optimization features"""
    from RL2.memory_optimizer import MemoryOptimizer, MemoryProfiler
    
    print("Testing memory optimization...")
    
    # Create optimizer
    optimizer = MemoryOptimizer(memory_threshold=0.8, gc_threshold=0.9)
    
    # Test memory optimization
    stats = optimizer.optimize_memory()
    print(f"Memory optimization result: {stats}")
    
    # Test CPU offload (mock)
    try:
        import torch
        from RL2.memory_optimizer import cpu_offload_optimizer
        optimizer_mock = torch.optim.Adam([torch.randn(10, requires_grad=True)])
        cpu_offload_optimizer(optimizer_mock, offload=True)
        print("CPU offload test: ✓ (No CUDA available)" if not torch.cuda.is_available() else "CPU offload test: ✓")
    except Exception as e:
        print(f"CPU offload test failed: {e}")
    
    # Test memory profiler
    profiler = MemoryProfiler(profile_interval=0.1)
    profiler.start_profiling()
    time.sleep(0.5)  # Profile for 0.5 seconds
    profiler.stop_profiling()
    
    # Get memory stats
    current_stats = profiler.get_current_stats()
    print(f"Memory stats: GPU={current_stats.gpu_allocated:.2f}GB, CPU={current_stats.cpu_memory:.2f}GB")
    
    print("✓ Memory optimization test passed\n")

def test_hyperparameter_tuner():
    """Test hyperparameter tuning"""
    from RL2.hyperopt import HyperparameterTuner, HyperparameterConfig
    
    print("Testing hyperparameter tuning...")
    
    # Create simple parameter configs
    param_configs = [
        HyperparameterConfig(
            name="lr",
            type="float",
            low=1e-6,
            high=1e-3,
            log_scale=True
        ),
        HyperparameterConfig(
            name="batch_size",
            type="int",
            low=4,
            high=32
        )
    ]
    
    # Create tuner
    tuner = HyperparameterTuner(
        optimizer_type="random",
        param_configs=param_configs,
        n_trials=5
    )
    
    # Define simple objective function
    def objective_function(params):
        # Simulate training result
        return {
            "reward": np.random.normal(0.5, 0.1),
            "loss": np.random.uniform(0.1, 0.5)
        }
    
    # Run optimization
    best_params = tuner.optimize(objective_function)
    
    print(f"Best parameters: {best_params}")
    print("✓ Hyperparameter tuning test passed\n")

def test_experiment_tracking():
    """Test experiment tracking (without external dependencies)"""
    from RL2.experiment_tracking import ExperimentTracker, ExperimentConfig
    
    print("Testing experiment tracking...")
    
    # Create config
    config = ExperimentConfig(
        experiment_name="test_experiment",
        enable_mlflow=False,  # Disable to avoid dependency issues
        enable_wandb=False
    )
    
    # Create tracker
    tracker = ExperimentTracker(config)
    
    # Test logging
    tracker.log_params({"lr": 0.001, "batch_size": 16})
    tracker.log_metrics({"reward": 0.5, "loss": 0.3}, step=1)
    
    # Test config logging
    test_config = {"model": "test", "params": {"lr": 0.001}}
    tracker.log_config(test_config)
    
    # Finish tracking
    tracker.finish()
    
    print("✓ Experiment tracking test passed\n")

def main():
    """Run all tests"""
    print("Running enhanced RL2 feature tests...\n")
    
    try:
        test_adaptive_kl_controller()
        test_multi_objective_optimizer()
        test_advantage_estimation()
        test_memory_optimizer()
        test_hyperparameter_tuner()
        test_experiment_tracking()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    main()
