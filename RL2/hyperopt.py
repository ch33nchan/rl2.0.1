# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import torch
from typing import Dict, List, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import time
import random
from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class HyperparameterConfig:
    """Configuration for a hyperparameter"""
    name: str
    type: str  # 'float', 'int', 'categorical', 'bool'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    default: Optional[Any] = None

class HyperparameterOptimizer(ABC):
    """Base class for hyperparameter optimization algorithms"""
    
    def __init__(self, 
                 param_configs: List[HyperparameterConfig],
                 objective_name: str = "reward",
                 maximize: bool = True,
                 n_trials: int = 100,
                 random_seed: int = 42):
        
        self.param_configs = {config.name: config for config in param_configs}
        self.objective_name = objective_name
        self.maximize = maximize
        self.n_trials = n_trials
        self.random_seed = random_seed
        
        # Trial history
        self.trials = []
        self.best_params = None
        self.best_value = float('-inf') if maximize else float('inf')
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    @abstractmethod
    def suggest_params(self, trial_id: int) -> Dict[str, Any]:
        """Suggest next set of parameters to try"""
        pass
    
    def update_trial(self, params: Dict[str, Any], results: Dict[str, float]):
        """Update with trial results"""
        objective_value = results.get(self.objective_name, 0.0)
        
        trial = {
            'trial_id': len(self.trials),
            'params': params.copy(),
            'results': results.copy(),
            'objective_value': objective_value,
            'timestamp': time.time()
        }
        
        self.trials.append(trial)
        
        # Update best
        is_better = (
            (self.maximize and objective_value > self.best_value) or
            (not self.maximize and objective_value < self.best_value)
        )
        
        if is_better:
            self.best_value = objective_value
            self.best_params = params.copy()
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found so far"""
        return self.best_params or {}
    
    def get_trial_history(self) -> List[Dict[str, Any]]:
        """Get complete trial history"""
        return self.trials
    
    def save_history(self, filepath: str):
        """Save optimization history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.trials, f, indent=2)
    
    def load_history(self, filepath: str):
        """Load optimization history from file"""
        with open(filepath, 'r') as f:
            self.trials = json.load(f)
        
        # Update best
        for trial in self.trials:
            if self.best_params is None:
                self.best_params = trial['params']
                self.best_value = trial['objective_value']
            else:
                is_better = (
                    (self.maximize and trial['objective_value'] > self.best_value) or
                    (not self.maximize and trial['objective_value'] < self.best_value)
                )
                if is_better:
                    self.best_value = trial['objective_value']
                    self.best_params = trial['params']

class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimization"""
    
    def suggest_params(self, trial_id: int) -> Dict[str, Any]:
        """Suggest random parameters"""
        params = {}
        
        for name, config in self.param_configs.items():
            if config.type == 'float':
                if config.log_scale:
                    log_low = np.log(config.low)
                    log_high = np.log(config.high)
                    params[name] = np.exp(np.random.uniform(log_low, log_high))
                else:
                    params[name] = np.random.uniform(config.low, config.high)
            elif config.type == 'int':
                params[name] = np.random.randint(config.low, config.high + 1)
            elif config.type == 'categorical':
                params[name] = np.random.choice(config.choices)
            elif config.type == 'bool':
                params[name] = np.random.choice([True, False])
        
        return params

class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search hyperparameter optimization"""
    
    def __init__(self, *args, grid_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size
        self.grid_points = self._generate_grid()
        self.grid_index = 0
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate grid of parameter combinations"""
        grid_points = []
        
        # Create grid for each parameter
        param_grids = {}
        for name, config in self.param_configs.items():
            if config.type == 'float':
                if config.log_scale:
                    log_low = np.log(config.low)
                    log_high = np.log(config.high)
                    grid_values = np.exp(np.linspace(log_low, log_high, self.grid_size))
                else:
                    grid_values = np.linspace(config.low, config.high, self.grid_size)
            elif config.type == 'int':
                grid_values = np.linspace(config.low, config.high, self.grid_size, dtype=int)
            elif config.type == 'categorical':
                grid_values = config.choices
            elif config.type == 'bool':
                grid_values = [True, False]
            
            param_grids[name] = grid_values
        
        # Generate all combinations
        param_names = list(param_grids.keys())
        if len(param_names) == 0:
            return []
        
        def generate_combinations(param_idx):
            if param_idx == len(param_names):
                return [{}]
            
            param_name = param_names[param_idx]
            param_values = param_grids[param_name]
            combinations = []
            
            for value in param_values:
                sub_combinations = generate_combinations(param_idx + 1)
                for sub_combo in sub_combinations:
                    combo = {param_name: value}
                    combo.update(sub_combo)
                    combinations.append(combo)
            
            return combinations
        
        return generate_combinations(0)
    
    def suggest_params(self, trial_id: int) -> Dict[str, Any]:
        """Suggest next grid point"""
        if self.grid_index >= len(self.grid_points):
            # Grid exhausted, return random
            return RandomSearchOptimizer.suggest_params(self, trial_id)
        
        params = self.grid_points[self.grid_index]
        self.grid_index += 1
        return params

class BayesianOptimizer(HyperparameterOptimizer):
    """Bayesian optimization using Gaussian Process (simplified implementation)"""
    
    def __init__(self, *args, exploration_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_weight = exploration_weight
        self.gp_X = []
        self.gp_y = []
    
    def suggest_params(self, trial_id: int) -> Dict[str, Any]:
        """Suggest parameters using acquisition function"""
        if len(self.trials) < 5:  # Cold start with random
            return RandomSearchOptimizer.suggest_params(self, trial_id)
        
        # Update GP with trial data
        self._update_gp()
        
        # Optimize acquisition function
        best_params = None
        best_acquisition = float('-inf')
        
        # Sample candidates and evaluate acquisition
        for _ in range(1000):  # Monte Carlo sampling
            candidate = RandomSearchOptimizer.suggest_params(self, trial_id)
            acquisition = self._acquisition_function(candidate)
            
            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_params = candidate
        
        return best_params or RandomSearchOptimizer.suggest_params(self, trial_id)
    
    def _update_gp(self):
        """Update Gaussian Process with trial data"""
        self.gp_X = []
        self.gp_y = []
        
        for trial in self.trials:
            # Convert params to vector
            x = self._params_to_vector(trial['params'])
            y = trial['objective_value']
            
            self.gp_X.append(x)
            self.gp_y.append(y)
    
    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numeric vector"""
        vector = []
        
        for name, config in self.param_configs.items():
            value = params[name]
            
            if config.type == 'float':
                if config.log_scale:
                    normalized = (np.log(value) - np.log(config.low)) / (np.log(config.high) - np.log(config.low))
                else:
                    normalized = (value - config.low) / (config.high - config.low)
                vector.append(normalized)
            elif config.type == 'int':
                normalized = (value - config.low) / (config.high - config.low)
                vector.append(normalized)
            elif config.type == 'categorical':
                # One-hot encoding
                for choice in config.choices:
                    vector.append(1.0 if value == choice else 0.0)
            elif config.type == 'bool':
                vector.append(1.0 if value else 0.0)
        
        return np.array(vector)
    
    def _acquisition_function(self, params: Dict[str, Any]) -> float:
        """Upper confidence bound acquisition function"""
        if not self.gp_X:
            return 0.0
        
        x = self._params_to_vector(params)
        
        # Simplified GP prediction (mean and variance)
        mean, variance = self._gp_predict(x)
        
        # UCB acquisition
        std = np.sqrt(variance)
        if self.maximize:
            return mean + self.exploration_weight * std
        else:
            return -(mean - self.exploration_weight * std)
    
    def _gp_predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Simplified GP prediction"""
        if not self.gp_X:
            return 0.0, 1.0
        
        # Simple distance-based prediction
        distances = [np.linalg.norm(x - gp_x) for gp_x in self.gp_X]
        weights = [np.exp(-d) for d in distances]
        weight_sum = sum(weights)
        
        if weight_sum == 0:
            return 0.0, 1.0
        
        # Weighted mean
        weighted_mean = sum(w * y for w, y in zip(weights, self.gp_y)) / weight_sum
        
        # Simple variance estimate
        variance = max(0.01, 1.0 / weight_sum)
        
        return weighted_mean, variance

class HyperparameterTuner:
    """Main hyperparameter tuning interface"""
    
    def __init__(self, 
                 optimizer_type: str = "random",
                 param_configs: List[HyperparameterConfig] = None,
                 objective_name: str = "reward",
                 maximize: bool = True,
                 n_trials: int = 100,
                 save_dir: str = "hyperopt_results",
                 **optimizer_kwargs):
        
        self.optimizer_type = optimizer_type
        self.param_configs = param_configs or []
        self.objective_name = objective_name
        self.maximize = maximize
        self.n_trials = n_trials
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create optimizer
        if optimizer_type == "random":
            self.optimizer = RandomSearchOptimizer(
                param_configs, objective_name, maximize, n_trials, **optimizer_kwargs
            )
        elif optimizer_type == "grid":
            self.optimizer = GridSearchOptimizer(
                param_configs, objective_name, maximize, n_trials, **optimizer_kwargs
            )
        elif optimizer_type == "bayesian":
            self.optimizer = BayesianOptimizer(
                param_configs, objective_name, maximize, n_trials, **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, train_function: Callable[[Dict[str, Any]], Dict[str, float]]) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        self.logger.info(f"Starting hyperparameter optimization with {self.optimizer_type} optimizer")
        self.logger.info(f"Target trials: {self.n_trials}, Objective: {self.objective_name}")
        
        for trial_id in range(self.n_trials):
            # Get suggested parameters
            params = self.optimizer.suggest_params(trial_id)
            
            self.logger.info(f"Trial {trial_id + 1}/{self.n_trials}: {params}")
            
            try:
                # Run training with suggested parameters
                results = train_function(params)
                
                # Update optimizer
                self.optimizer.update_trial(params, results)
                
                # Log results
                objective_value = results.get(self.objective_name, 0.0)
                self.logger.info(f"Trial {trial_id + 1} result: {self.objective_name}={objective_value:.4f}")
                
                # Save intermediate results
                if (trial_id + 1) % 10 == 0:
                    self.save_results()
                
            except Exception as e:
                self.logger.error(f"Trial {trial_id + 1} failed: {e}")
                # Continue with next trial
                continue
        
        # Save final results
        self.save_results()
        
        best_params = self.optimizer.get_best_params()
        self.logger.info(f"Optimization complete. Best params: {best_params}")
        self.logger.info(f"Best {self.objective_name}: {self.optimizer.best_value:.4f}")
        
        return best_params
    
    def save_results(self):
        """Save optimization results"""
        # Save trial history
        history_file = self.save_dir / "optimization_history.json"
        self.optimizer.save_history(str(history_file))
        
        # Save best parameters
        best_params_file = self.save_dir / "best_params.json"
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_params': self.optimizer.get_best_params(),
                'best_value': self.optimizer.best_value,
                'objective_name': self.objective_name,
                'n_trials': len(self.optimizer.trials)
            }, f, indent=2)
        
        # Save summary statistics
        self.save_summary_stats()
    
    def save_summary_stats(self):
        """Save summary statistics"""
        trials = self.optimizer.get_trial_history()
        if not trials:
            return
        
        # Compute statistics
        objective_values = [trial['objective_value'] for trial in trials]
        
        stats = {
            'n_trials': len(trials),
            'objective_name': self.objective_name,
            'best_value': self.optimizer.best_value,
            'mean_objective': np.mean(objective_values),
            'std_objective': np.std(objective_values),
            'median_objective': np.median(objective_values),
            'improvement_over_trials': self._compute_improvement_curve(trials)
        }
        
        # Save stats
        stats_file = self.save_dir / "optimization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _compute_improvement_curve(self, trials: List[Dict[str, Any]]) -> List[float]:
        """Compute improvement curve over trials"""
        best_so_far = []
        current_best = float('-inf') if self.maximize else float('inf')
        
        for trial in trials:
            objective_value = trial['objective_value']
            
            if self.maximize:
                current_best = max(current_best, objective_value)
            else:
                current_best = min(current_best, objective_value)
            
            best_so_far.append(current_best)
        
        return best_so_far

def create_ppo_hyperparameter_configs() -> List[HyperparameterConfig]:
    """Create default hyperparameter configurations for PPO"""
    return [
        HyperparameterConfig(
            name="actor.lr",
            type="float",
            low=1e-6,
            high=1e-3,
            log_scale=True,
            default=1e-5
        ),
        HyperparameterConfig(
            name="critic.lr",
            type="float",
            low=1e-6,
            high=1e-3,
            log_scale=True,
            default=5e-6
        ),
        HyperparameterConfig(
            name="actor.clip",
            type="float",
            low=0.1,
            high=0.5,
            default=0.2
        ),
        HyperparameterConfig(
            name="critic.clip",
            type="float",
            low=0.1,
            high=1.0,
            default=0.5
        ),
        HyperparameterConfig(
            name="adv.gamma",
            type="float",
            low=0.9,
            high=1.0,
            default=0.99
        ),
        HyperparameterConfig(
            name="adv.lamda",
            type="float",
            low=0.8,
            high=1.0,
            default=0.95
        ),
        HyperparameterConfig(
            name="actor.kl.coef",
            type="float",
            low=0.001,
            high=1.0,
            log_scale=True,
            default=0.1
        ),
        HyperparameterConfig(
            name="actor.entropy.coef",
            type="float",
            low=0.0,
            high=0.1,
            default=0.01
        ),
        HyperparameterConfig(
            name="adv.estimator",
            type="categorical",
            choices=["gae", "reinforce", "vtrace", "retrace", "td_lambda"],
            default="gae"
        ),
        HyperparameterConfig(
            name="rollout.train_sampling_params.temperature",
            type="float",
            low=0.1,
            high=2.0,
            default=1.0
        )
    ]

def update_config_with_params(config: OmegaConf, params: Dict[str, Any]) -> OmegaConf:
    """Update Hydra config with hyperparameter values"""
    config = OmegaConf.create(config)
    
    for param_name, value in params.items():
        # Handle nested parameter names like "actor.lr"
        keys = param_name.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    return config
