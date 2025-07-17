#!/usr/bin/env python3
"""
Example: Memory Optimizer
"""
from RL2.memory_optimizer import MemoryOptimizer, get_memory_summary
import torch

if __name__ == "__main__":
    optimizer = MemoryOptimizer(memory_threshold=0.8, gc_threshold=0.9)
    # Simulate memory usage
    tensors = [torch.randn(1000, 1000) for _ in range(5)]
    optimizer.optimize_memory()
    summary = get_memory_summary()
    print(f"Memory Summary: {summary}")
