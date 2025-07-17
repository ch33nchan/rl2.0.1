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

import os
import json
import time
import torch
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from omegaconf import OmegaConf

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    tracking_uri: Optional[str] = None
    experiment_name: str = "RL2_Experiment"
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    enable_mlflow: bool = True
    enable_wandb: bool = True
    log_models: bool = True
    log_artifacts: bool = True
    log_system_metrics: bool = True
    auto_log: bool = True

class ExperimentTracker:
    """Unified experiment tracking interface supporting MLflow and W&B"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Track active runs
        self.mlflow_run = None
        self.wandb_run = None
        
        # Metrics storage
        self.metrics_buffer = []
        self.step_counter = 0
        
        # Initialize trackers
        self.initialize_trackers()
    
    def initialize_trackers(self):
        """Initialize experiment tracking backends"""
        
        # Initialize MLflow
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            self.initialize_mlflow()
        elif self.config.enable_mlflow and not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not available. Install with: pip install mlflow")
        
        # Initialize W&B
        if self.config.enable_wandb and WANDB_AVAILABLE:
            self.initialize_wandb()
        elif self.config.enable_wandb and not WANDB_AVAILABLE:
            self.logger.warning("W&B not available. Install with: pip install wandb")
    
    def initialize_mlflow(self):
        """Initialize MLflow tracking"""
        try:
            # Set tracking URI
            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Set experiment
            mlflow.set_experiment(self.config.experiment_name)
            
            # Start run
            self.mlflow_run = mlflow.start_run(
                run_name=self.config.run_name,
                tags=self.config.tags or {}
            )
            
            # Enable auto-logging
            if self.config.auto_log:
                mlflow.pytorch.autolog()
            
            self.logger.info(f"MLflow initialized. Run ID: {self.mlflow_run.info.run_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow: {e}")
            self.mlflow_run = None
    
    def initialize_wandb(self):
        """Initialize W&B tracking"""
        try:
            # Initialize W&B
            self.wandb_run = wandb.init(
                project=self.config.experiment_name,
                name=self.config.run_name,
                tags=list(self.config.tags.values()) if self.config.tags else None,
                config=self.config.tags or {}
            )
            
            self.logger.info(f"W&B initialized. Run ID: {self.wandb_run.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {e}")
            self.wandb_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        # MLflow
        if self.mlflow_run:
            try:
                mlflow.log_params(params)
            except Exception as e:
                self.logger.error(f"Failed to log params to MLflow: {e}")
        
        # W&B
        if self.wandb_run:
            try:
                wandb.config.update(params)
            except Exception as e:
                self.logger.error(f"Failed to log params to W&B: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Add to buffer
        self.metrics_buffer.append({
            'metrics': metrics.copy(),
            'step': step,
            'timestamp': time.time()
        })
        
        # MLflow
        if self.mlflow_run:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                self.logger.error(f"Failed to log metrics to MLflow: {e}")
        
        # W&B
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                self.logger.error(f"Failed to log metrics to W&B: {e}")
    
    def log_model(self, model: torch.nn.Module, model_name: str, step: Optional[int] = None):
        """Log model artifacts"""
        if not self.config.log_models:
            return
        
        # MLflow
        if self.mlflow_run:
            try:
                mlflow.pytorch.log_model(
                    model, 
                    model_name,
                    registered_model_name=f"{self.config.experiment_name}_{model_name}"
                )
            except Exception as e:
                self.logger.error(f"Failed to log model to MLflow: {e}")
        
        # W&B
        if self.wandb_run:
            try:
                # Save model to temporary file
                temp_path = f"/tmp/{model_name}_{step or 0}.pth"
                torch.save(model.state_dict(), temp_path)
                
                # Log as artifact
                wandb.save(temp_path, base_path="/tmp")
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                self.logger.error(f"Failed to log model to W&B: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact files"""
        if not self.config.log_artifacts:
            return
        
        # MLflow
        if self.mlflow_run:
            try:
                mlflow.log_artifact(artifact_path, artifact_name)
            except Exception as e:
                self.logger.error(f"Failed to log artifact to MLflow: {e}")
        
        # W&B
        if self.wandb_run:
            try:
                wandb.save(artifact_path)
            except Exception as e:
                self.logger.error(f"Failed to log artifact to W&B: {e}")
    
    def log_text(self, text: str, filename: str):
        """Log text content"""
        # Create temporary file
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'w') as f:
            f.write(text)
        
        # Log as artifact
        self.log_artifact(temp_path, filename)
        
        # Clean up
        os.remove(temp_path)
    
    def log_table(self, data: List[Dict[str, Any]], table_name: str):
        """Log tabular data"""
        # W&B
        if self.wandb_run:
            try:
                table = wandb.Table(data=data)
                wandb.log({table_name: table})
            except Exception as e:
                self.logger.error(f"Failed to log table to W&B: {e}")
        
        # MLflow (as JSON)
        if self.mlflow_run:
            try:
                json_data = json.dumps(data, indent=2)
                self.log_text(json_data, f"{table_name}.json")
            except Exception as e:
                self.logger.error(f"Failed to log table to MLflow: {e}")
    
    def log_config(self, config: Union[Dict[str, Any], OmegaConf]):
        """Log configuration"""
        if isinstance(config, OmegaConf):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
        
        # Log as parameters
        self.log_params(self.flatten_dict(config_dict))
        
        # Log as artifact
        config_json = json.dumps(config_dict, indent=2, default=str)
        self.log_text(config_json, "config.json")
    
    def flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_system_metrics(self):
        """Log system metrics"""
        if not self.config.log_system_metrics:
            return
        
        try:
            import psutil
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = {
                'system/cpu_percent': cpu_percent,
                'system/memory_percent': memory.percent,
                'system/memory_available_gb': memory.available / (1024**3)
            }
            
            # GPU metrics
            if torch.cuda.is_available():
                metrics.update({
                    'system/gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'system/gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'system/gpu_utilization': torch.cuda.utilization()
                })
            
            self.log_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to log system metrics: {e}")
    
    def finish(self):
        """Finish experiment tracking"""
        # Save metrics buffer
        if self.metrics_buffer:
            self.log_artifact_from_data(self.metrics_buffer, "metrics_history.json")
        
        # End MLflow run
        if self.mlflow_run:
            try:
                mlflow.end_run()
                self.logger.info("MLflow run ended")
            except Exception as e:
                self.logger.error(f"Failed to end MLflow run: {e}")
        
        # End W&B run
        if self.wandb_run:
            try:
                wandb.finish()
                self.logger.info("W&B run ended")
            except Exception as e:
                self.logger.error(f"Failed to end W&B run: {e}")
    
    def log_artifact_from_data(self, data: Any, filename: str):
        """Log data as artifact"""
        temp_path = f"/tmp/{filename}"
        
        # Save data
        if filename.endswith('.json'):
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif filename.endswith('.pkl'):
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Try JSON by default
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Log as artifact
        self.log_artifact(temp_path, filename)
        
        # Clean up
        os.remove(temp_path)
    
    @contextmanager
    def experiment_context(self):
        """Context manager for experiment tracking"""
        try:
            yield self
        finally:
            self.finish()

class ModelVersioning:
    """Model versioning and registry integration"""
    
    def __init__(self, 
                 experiment_tracker: ExperimentTracker,
                 model_registry_uri: Optional[str] = None,
                 enable_auto_versioning: bool = True):
        
        self.tracker = experiment_tracker
        self.model_registry_uri = model_registry_uri
        self.enable_auto_versioning = enable_auto_versioning
        
        # Version tracking
        self.model_versions = {}
        self.version_counter = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model registry
        if self.model_registry_uri and MLFLOW_AVAILABLE:
            mlflow.set_registry_uri(self.model_registry_uri)
    
    def register_model(self, 
                      model: torch.nn.Module,
                      model_name: str,
                      version: Optional[str] = None,
                      stage: str = "None",
                      description: str = "",
                      tags: Optional[Dict[str, str]] = None) -> str:
        """Register model in registry"""
        
        if version is None:
            version = self.get_next_version(model_name)
        
        # Log model with tracker
        self.tracker.log_model(model, model_name)
        
        # Register in MLflow registry
        if MLFLOW_AVAILABLE and self.tracker.mlflow_run:
            try:
                model_uri = f"runs:/{self.tracker.mlflow_run.info.run_id}/{model_name}"
                
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                    tags=tags or {}
                )
                
                # Update version info
                self.model_versions[model_name] = version
                
                self.logger.info(f"Model {model_name} registered with version {version}")
                
                return registered_model.version
                
            except Exception as e:
                self.logger.error(f"Failed to register model: {e}")
                return version
        
        return version
    
    def get_next_version(self, model_name: str) -> str:
        """Get next version number for model"""
        if not self.enable_auto_versioning:
            return "1.0.0"
        
        self.version_counter += 1
        return f"1.0.{self.version_counter}"
    
    def load_model(self, model_name: str, version: str = "latest") -> torch.nn.Module:
        """Load model from registry"""
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow not available for model loading")
        
        try:
            if version == "latest":
                model_uri = f"models:/{model_name}/latest"
            else:
                model_uri = f"models:/{model_name}/{version}"
            
            model = mlflow.pytorch.load_model(model_uri)
            
            self.logger.info(f"Model {model_name} (version {version}) loaded from registry")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e
    
    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to different stage"""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            self.logger.info(f"Model {model_name} v{version} promoted to {stage}")
            
        except Exception as e:
            self.logger.error(f"Failed to promote model: {e}")
    
    def compare_models(self, model_name: str, versions: List[str]) -> Dict[str, Any]:
        """Compare model versions"""
        if not MLFLOW_AVAILABLE:
            return {}
        
        try:
            client = mlflow.tracking.MlflowClient()
            comparison = {}
            
            for version in versions:
                model_version = client.get_model_version(model_name, version)
                run = client.get_run(model_version.run_id)
                
                comparison[version] = {
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'creation_time': model_version.creation_timestamp,
                    'stage': model_version.current_stage
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare models: {e}")
            return {}

def create_experiment_tracker(config: Dict[str, Any]) -> ExperimentTracker:
    """Create experiment tracker from config"""
    exp_config = ExperimentConfig(**config)
    return ExperimentTracker(exp_config)

def setup_tracking_for_ppo(config: OmegaConf, 
                           experiment_name: str,
                           run_name: Optional[str] = None) -> ExperimentTracker:
    """Setup experiment tracking for PPO training"""
    
    # Create tracker config
    tracker_config = ExperimentConfig(
        experiment_name=experiment_name,
        run_name=run_name or f"ppo_{int(time.time())}",
        tags={
            "algorithm": "PPO",
            "model": config.actor.model_name,
            "advantage_estimator": config.adv.estimator
        }
    )
    
    # Create tracker
    tracker = ExperimentTracker(tracker_config)
    
    # Log configuration
    tracker.log_config(config)
    
    return tracker
