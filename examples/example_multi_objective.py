#!/usr/bin/env python3
"""
Example: Multi-Objective Optimizer
"""
from RL2.algs import MultiObjectiveOptimizer

if __name__ == "__main__":
    mo_optimizer = MultiObjectiveOptimizer(
        objectives=['reward', 'kl_penalty', 'entropy'],
        objective_weights={'reward': 1.0, 'kl_penalty': 0.5, 'entropy': 0.1},
        pareto_method='weighted_sum',
        diversity_weight=0.0,
        archive_size=10
    )
    # Simulate some objective values
    for i in range(5):
        objectives = {
            'reward': 1.0 + 0.1 * i,
            'kl_penalty': -0.01 * i,
            'entropy': 0.05 + 0.01 * i
        }
        loss = mo_optimizer.compute_pareto_loss(objectives)
        mo_optimizer.update_pareto_archive(objectives)
        print(f"Step {i}: Objectives={objectives}, Pareto Loss={loss:.4f}")
    print(f"Pareto Archive Size: {len(mo_optimizer.pareto_archive)}")
