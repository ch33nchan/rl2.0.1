#!/usr/bin/env python3
"""
Enhanced RL2 Example with All New Features
==========================================

# Example Usage (Bash)
# --------------------
# To run individual component examples:
#
#   python examples/example_adaptive_kl.py
#   python examples/example_multi_objective.py
#   python examples/example_memory_optimizer.py
#   python examples/example_experiment_tracker.py
#
# To run all examples in sequence:
#
#   bash examples/run_all_examples.sh
#
# To run the full enhanced RL2 demo:
#
#   python examples/enhanced_rl2_example.py --config examples/enhanced_ppo_config.yaml
#

This script demonstrates the new features added to RL2:
1. Adaptive KL penalty mechanisms
2. Multi-objective optimization with Pareto frontiers
3. Alternative advantage estimation methods
4. Automated hyperparameter tuning
5. Advanced memory optimization
6. Adaptive batch sizing
7. Experiment tracking with MLflow/W&B
8. Model versioning

Usage:
    python enhanced_rl2_example.py --config examples/enhanced_ppo_config.yaml
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

# Add RL2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from RL2.algs import (
    AdaptiveKLController, 
    ConstraintOptimizer, 
    MultiObjectiveOptimizer,
    compute_advantages_unified
)
from RL2.hyperopt import (
    HyperparameterTuner,
    HyperparameterConfig,
    create_ppo_hyperparameter_configs,
    update_config_with_params
)
from RL2.memory_optimizer import (
    MemoryOptimizer,
    MemoryProfiler,
    AdaptiveBatchSizer,
    memory_profiling
)
from RL2.experiment_tracking import (
    ExperimentTracker,
    ExperimentConfig,
    ModelVersioning,
    setup_tracking_for_ppo
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedPPOTrainer:
    """Enhanced PPO trainer with all new features"""
    
    def __init__(self, config: OmegaConf):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.experiment_tracker = None
        self.kl_controller = None
        self.constraint_optimizer = None
        self.multi_objective_optimizer = None
        self.memory_optimizer = None
        self.memory_profiler = None
        self.adaptive_batch_sizer = None
        self.model_versioning = None
        
        # Initialize enhanced components
        self.setup_enhanced_components()
    
    def setup_enhanced_components(self):
        """Setup all enhanced components"""
        
        # Experiment tracking
        if self.config.experiment_tracking.enabled:
            self.experiment_tracker = setup_tracking_for_ppo(
                self.config, 
                self.config.experiment_tracking.experiment_name,
                self.config.experiment_tracking.run_name
            )
            
            # Model versioning
            if self.config.experiment_tracking.model_versioning.enabled:
                self.model_versioning = ModelVersioning(
                    self.experiment_tracker,
                    self.config.experiment_tracking.model_versioning.model_registry_uri,
                    self.config.experiment_tracking.model_versioning.enable_auto_versioning
                )
        
        # Adaptive KL penalty
        if self.config.actor.kl.adaptive_kl.enabled:
            self.kl_controller = AdaptiveKLController(
                initial_coef=self.config.actor.kl.adaptive_kl.initial_coef,
                target_kl=self.config.actor.kl.adaptive_kl.target_kl,
                controller_type=self.config.actor.kl.adaptive_kl.controller_type,
                exp_factor=self.config.actor.kl.adaptive_kl.exp_factor,
                exp_decay=self.config.actor.kl.adaptive_kl.exp_decay,
                linear_factor=self.config.actor.kl.adaptive_kl.linear_factor,
                kp=self.config.actor.kl.adaptive_kl.kp,
                ki=self.config.actor.kl.adaptive_kl.ki,
                kd=self.config.actor.kl.adaptive_kl.kd,
                schedule_type=self.config.actor.kl.adaptive_kl.schedule_type,
                schedule_steps=self.config.actor.kl.adaptive_kl.schedule_steps,
                min_coef=self.config.actor.kl.adaptive_kl.min_coef,
                max_coef=self.config.actor.kl.adaptive_kl.max_coef
            )
        
        # Constraint optimization
        if self.config.actor.kl.constraint_optimization.enabled:
            self.constraint_optimizer = ConstraintOptimizer(
                kl_constraint=self.config.actor.kl.constraint_optimization.kl_constraint,
                entropy_constraint=self.config.actor.kl.constraint_optimization.entropy_constraint,
                use_lagrangian=self.config.actor.kl.constraint_optimization.use_lagrangian,
                lagrangian_lr=self.config.actor.kl.constraint_optimization.lagrangian_lr,
                constraint_violation_penalty=self.config.actor.kl.constraint_optimization.constraint_violation_penalty
            )
        
        # Multi-objective optimization
        if self.config.multi_objective.enabled:
            self.multi_objective_optimizer = MultiObjectiveOptimizer(
                objectives=self.config.multi_objective.objectives,
                objective_weights=dict(self.config.multi_objective.objective_weights),
                pareto_method=self.config.multi_objective.pareto_method,
                diversity_weight=self.config.multi_objective.diversity_weight,
                archive_size=self.config.multi_objective.archive_size
            )
        
        # Memory optimization
        if self.config.memory_optimization.enabled:
            self.memory_optimizer = MemoryOptimizer(
                memory_threshold=self.config.memory_optimization.memory_threshold,
                gc_threshold=self.config.memory_optimization.gc_threshold
            )
            
            # Memory profiler
            if self.config.memory_optimization.profiling.enabled:
                self.memory_profiler = MemoryProfiler(
                    profile_interval=self.config.memory_optimization.profiling.profile_interval,
                    save_path=self.config.memory_optimization.profiling.save_path
                )
            
            # Adaptive batch sizing
            if self.config.memory_optimization.adaptive_batch_sizing.enabled:
                self.adaptive_batch_sizer = AdaptiveBatchSizer(
                    initial_batch_size=self.config.memory_optimization.adaptive_batch_sizing.initial_batch_size,
                    min_batch_size=self.config.memory_optimization.adaptive_batch_sizing.min_batch_size,
                    max_batch_size=self.config.memory_optimization.adaptive_batch_sizing.max_batch_size,
                    adaptation_factor=self.config.memory_optimization.adaptive_batch_sizing.adaptation_factor
                )
    
    def train_step(self, data_list: List[Dict[str, Any]], step: int) -> Dict[str, float]:
        """Enhanced training step with all new features"""
        
        # Memory profiling context
        context_manager = self.memory_profiler.profile_context(f"train_step_{step}") if self.memory_profiler else None
        
        if context_manager:
            with context_manager:
                return self._train_step_impl(data_list, step)
        else:
            return self._train_step_impl(data_list, step)
    
    def _train_step_impl(self, data_list: List[Dict[str, Any]], step: int) -> Dict[str, float]:
        """Implementation of training step"""
        
        # Compute advantages with selected method
        compute_advantages_unified(
            data_list, 
            method=self.config.adv.estimator,
            gamma=self.config.adv.gamma,
            lamda=self.config.adv.lamda,
            norm_var=self.config.adv.norm_var,
            # Method-specific parameters
            rho_bar=self.config.adv.vtrace.rho_bar,
            c_bar=self.config.adv.vtrace.c_bar,
            clip_ratio=self.config.adv.clipped_is.clip_ratio,
            n_steps=self.config.adv.multistep.n_steps,
            bootstrap_method=self.config.adv.multistep.bootstrap_method
        )
        
        # Simulate training metrics
        kl_div = np.random.exponential(0.01)
        entropy = np.random.exponential(0.05)
        reward = np.random.normal(0.5, 0.1)
        
        # Update adaptive KL controller
        if self.kl_controller:
            kl_coef = self.kl_controller.update(kl_div, step)
            self.logger.info(f"Step {step}: KL coef updated to {kl_coef:.4f}")
        
        # Check constraints
        if self.constraint_optimizer:
            constraint_info = self.constraint_optimizer.check_constraints(kl_div, entropy)
            if not constraint_info["kl_satisfied"] or not constraint_info["entropy_satisfied"]:
                self.logger.warning(f"Step {step}: Constraint violation detected")
            
            # Update Lagrangian multipliers
            self.constraint_optimizer.update_lagrangian_multipliers(kl_div, entropy)
        
        # Multi-objective optimization
        if self.multi_objective_optimizer:
            objectives_dict = {
                "reward": reward,
                "kl_penalty": -kl_div,
                "entropy": entropy
            }
            
            multi_obj_loss = self.multi_objective_optimizer.compute_pareto_loss(objectives_dict)
            self.multi_objective_optimizer.update_pareto_archive(objectives_dict)
        
        # Adaptive batch sizing
        if self.adaptive_batch_sizer:
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0.5
            training_time = np.random.uniform(1.0, 3.0)  # Simulate training time
            success = np.random.random() > 0.1  # Simulate success rate
            
            new_batch_size = self.adaptive_batch_sizer.update_batch_size(
                memory_usage, training_time, success
            )
            
            if new_batch_size != self.adaptive_batch_sizer.current_batch_size:
                self.logger.info(f"Step {step}: Batch size updated to {new_batch_size}")
        
        # Memory optimization
        if self.memory_optimizer:
            self.memory_optimizer.optimize_memory()
        
        # Prepare metrics
        metrics = {
            "reward": reward,
            "kl_divergence": kl_div,
            "entropy": entropy,
            "loss": np.random.uniform(0.1, 0.5)
        }
        
        # Add component-specific metrics
        if self.kl_controller:
            metrics.update({
                "kl_coef": self.kl_controller.coef,
                "kl_target": self.kl_controller.target_kl
            })
        
        if self.constraint_optimizer:
            constraint_stats = self.constraint_optimizer.get_stats()
            metrics.update({
                "constraint_violations": constraint_stats["constraint_violations"],
                "kl_multiplier": constraint_stats["kl_multiplier"],
                "entropy_multiplier": constraint_stats["entropy_multiplier"]
            })
        
        if self.multi_objective_optimizer:
            mo_stats = self.multi_objective_optimizer.get_stats()
            metrics.update({
                "pareto_size": mo_stats["pareto_size"],
                "total_evaluations": mo_stats["total_evaluations"]
            })
        
        if self.adaptive_batch_sizer:
            batch_stats = self.adaptive_batch_sizer.get_stats()
            metrics.update({
                "current_batch_size": batch_stats["current_batch_size"],
                "success_rate": batch_stats["success_rate"]
            })
        
        if self.memory_optimizer:
            from RL2.memory_optimizer import get_memory_summary
            memory_stats = get_memory_summary()
            metrics.update({
                "memory_gpu_allocated": memory_stats.get("gpu", {}).get("memory_allocated", 0),
                "memory_cpu_percent": memory_stats.get("cpu", {}).get("memory_percent", 0)
            })
        
        # Log metrics
        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(metrics, step)
            
            # Log system metrics periodically
            if step % 10 == 0:
                self.experiment_tracker.log_system_metrics()
        
        return metrics
    
    def train(self, num_steps: int = 1000):
        """Main training loop"""
        
        self.logger.info(f"Starting enhanced PPO training for {num_steps} steps")
        
        try:
            for step in range(num_steps):
                # Simulate data
                data_list = [
                    {
                        "rewards": torch.randn(100),
                        "values": torch.randn(100),
                        "logps": torch.randn(100),
                        "ref_logps": torch.randn(100),
                        "action_mask": torch.randint(0, 2, (100,)).bool()
                    }
                    for _ in range(4)  # Simulate 4 sequences
                ]
                
                # Training step
                metrics = self.train_step(data_list, step)
                
                # Log progress
                if step % 100 == 0:
                    self.logger.info(f"Step {step}: reward={metrics['reward']:.4f}, "
                                   f"kl_div={metrics['kl_divergence']:.4f}, "
                                   f"entropy={metrics['entropy']:.4f}")
                
                # Save model periodically
                if step % 500 == 0 and self.model_versioning:
                    # Simulate model saving
                    dummy_model = torch.nn.Linear(10, 1)
                    version = self.model_versioning.register_model(
                        dummy_model,
                        "ppo_actor",
                        description=f"Model at step {step}"
                    )
                    self.logger.info(f"Model saved with version {version}")
        
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        
        # Stop memory profiler
        if self.memory_profiler:
            self.memory_profiler.stop_profiling()
        
        # Clear memory pools
        if self.memory_optimizer:
            from RL2.memory_optimizer import clear_memory_cache
            clear_memory_cache()
        
        # Finish experiment tracking
        if self.experiment_tracker:
            self.experiment_tracker.finish()
        
        self.logger.info("Training cleanup completed")

def run_hyperparameter_optimization(config: OmegaConf) -> Dict[str, Any]:
    """Run hyperparameter optimization"""
    
    logger.info("Starting hyperparameter optimization")
    
    # Create parameter configurations
    param_configs = create_ppo_hyperparameter_configs()
    
    # Create tuner
    tuner = HyperparameterTuner(
        optimizer_type=config.hyperopt.optimizer_type,
        param_configs=param_configs,
        objective_name="reward",
        maximize=True,
        n_trials=config.hyperopt.n_trials,
        save_dir=config.hyperopt.save_dir,
        exploration_weight=config.hyperopt.bayesian.exploration_weight,
        grid_size=config.hyperopt.grid.grid_size
    )
    
    def train_function(params: Dict[str, Any]) -> Dict[str, float]:
        """Training function for hyperparameter optimization"""
        
        # Update config with suggested parameters
        updated_config = update_config_with_params(config, params)
        
        # Create trainer
        trainer = EnhancedPPOTrainer(updated_config)
        
        # Run short training
        trainer.train(num_steps=100)
        
        # Return metrics (simulate)
        return {
            "reward": np.random.normal(0.5, 0.1),
            "kl_divergence": np.random.exponential(0.01),
            "entropy": np.random.exponential(0.05)
        }
    
    # Run optimization
    best_params = tuner.optimize(train_function)
    
    logger.info(f"Hyperparameter optimization completed. Best params: {best_params}")
    
    return best_params

def demonstrate_memory_optimization():
    """Demonstrate memory optimization features"""
    
    logger.info("Demonstrating memory optimization features")
    
    # Create memory optimizer
    optimizer = MemoryOptimizer(memory_threshold=0.8, gc_threshold=0.9)
    
    # Create memory profiler
    profiler = MemoryProfiler(
        profile_interval=0.5,
        save_path="memory_demo_profile.json"
    )
    
    # Simulate memory-intensive operations
    with memory_profiling(profile_interval=0.1):
        # Simulate large tensor operations
        large_tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            
            large_tensors.append(tensor)
            
            # Simulate processing
            result = torch.mm(tensor, tensor.t())
            
            time.sleep(0.1)  # Simulate processing time
    
    # Get memory summary
    from RL2.memory_optimizer import get_memory_summary
    memory_summary = get_memory_summary()
    logger.info(f"Memory summary: {memory_summary}")
    
    # Stop profiler
    profiler.stop_profiling()
    
    logger.info("Memory optimization demonstration completed")

@hydra.main(version_base=None, config_path="../RL2/trainer/config", config_name="ppo")
def main(config: OmegaConf):
    """Main function demonstrating all enhanced features"""
    
    logger.info("Starting Enhanced RL2 demonstration")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    try:
        # Demonstrate memory optimization
        demonstrate_memory_optimization()
        
        # Run hyperparameter optimization if enabled
        if config.hyperopt.enabled:
            best_params = run_hyperparameter_optimization(config)
            
            # Update config with best parameters
            config = update_config_with_params(config, best_params)
        
        # Create enhanced trainer
        trainer = EnhancedPPOTrainer(config)
        
        # Run training
        trainer.train(num_steps=1000)
        
        logger.info("Enhanced RL2 demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()
