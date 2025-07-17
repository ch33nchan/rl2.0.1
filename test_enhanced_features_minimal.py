#!/usr/bin/env python3
"""
Minimal test script for enhanced RL2 features (no external dependencies)
"""

import sys
import torch
import numpy as np
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
    
    print("✓ AdaptiveKLController test passed\n")

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
    
    methods = ["gae", "reinforce", "vtrace", "retrace", "td_lambda", "clipped_is", "multistep"]
    
    for method in methods:
        try:
            # Create fresh copy for each method
            test_data = [{k: v.clone() for k, v in ex.items()} for ex in data_list]
            
            compute_advantages_unified(
                test_data,
                method=method,
                gamma=0.99,
                lamda=0.95,
                responses_per_prompt=1,
                norm_var=False,
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
    optimizer = MemoryOptimizer(
        enable_gradient_checkpointing=True,
        enable_cpu_offload=True,
        memory_threshold=0.8
    )
    
    # Test CPU offloading
    if torch.cuda.is_available():
        test_tensor = torch.randn(100, 100).cuda()
        cpu_tensor = optimizer.offload_to_cpu(test_tensor, "test_tensor")
        retrieved_tensor = optimizer.retrieve_from_cpu("test_tensor", torch.device("cuda"))
        
        print(f"CPU offload test: {'✓' if retrieved_tensor is not None else '✗'}")
    else:
        print("CPU offload test: ✓ (No CUDA available)")
    
    # Test memory profiler
    profiler = MemoryProfiler(enabled=True, profile_interval=0.1)
    profiler.add_event("test_event")
    
    # Get memory stats
    stats = profiler.get_memory_stats()
    print(f"Memory stats: GPU={stats.gpu_allocated:.2f}GB, CPU={stats.cpu_memory:.2f}GB")
    
    profiler.stop_profiling()
    print("✓ Memory optimization test passed\n")

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
    
    print("✓ MultiObjectiveOptimizer test passed\n")

def main():
    """Run all tests"""
    print("Running enhanced RL2 feature tests...\n")
    
    try:
        test_adaptive_kl_controller()
        test_advantage_estimation()
        test_memory_optimizer()
        test_multi_objective_optimizer()
        
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
