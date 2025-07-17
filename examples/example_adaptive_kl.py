#!/usr/bin/env python3
"""
Example: Adaptive KL Controller
"""
from RL2.algs import AdaptiveKLController

if __name__ == "__main__":
    kl_controller = AdaptiveKLController(
        controller_type='exponential',
        target_kl=0.01,
        initial_coef=0.2
    )
    print(f"Initial KL Coef: {kl_controller.coef}")
    for step in range(5):
        kl = 0.01 * (1 + 0.1 * step)
        coef = kl_controller.update(kl, step)
        print(f"Step {step}: KL={kl:.4f}, Updated Coef={coef:.4f}")
