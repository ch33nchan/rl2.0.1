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

import gc
import psutil
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time
import threading
import logging
from pathlib import Path
import json

@dataclass
class MemoryStats:
    """Memory statistics container"""
    gpu_allocated: float
    gpu_reserved: float
    gpu_max_allocated: float
    gpu_max_reserved: float
    cpu_memory: float
    cpu_percent: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class MemoryProfiler:
    """Memory profiling utility"""
    
    def __init__(self, profile_interval: float = 1.0, save_path: Optional[str] = None):
        self.profile_interval = profile_interval
        self.save_path = save_path
        self.profiling = False
        self.profile_data = []
        self.logger = logging.getLogger(__name__)
        self._profile_thread = None
        
    def start_profiling(self):
        """Start memory profiling"""
        if self.profiling:
            return
            
        self.profiling = True
        self.profile_data = []
        self.logger.info("Memory profiler started")
        
        def profile_loop():
            while self.profiling:
                stats = self._get_memory_stats()
                self.profile_data.append(stats)
                time.sleep(self.profile_interval)
        
        self._profile_thread = threading.Thread(target=profile_loop)
        self._profile_thread.daemon = True
        self._profile_thread.start()
        
    def stop_profiling(self):
        """Stop memory profiling"""
        if not self.profiling:
            return
            
        self.profiling = False
        if self._profile_thread:
            self._profile_thread.join()
            
        self.logger.info("Memory profiler stopped")
        
        if self.save_path:
            self.save_profile()
            
    def _get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            gpu_max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        else:
            gpu_allocated = gpu_reserved = gpu_max_allocated = gpu_max_reserved = 0.0
            
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3  # GB
        cpu_percent = process.memory_percent()
        
        return MemoryStats(
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            gpu_max_allocated=gpu_max_allocated,
            gpu_max_reserved=gpu_max_reserved,
            cpu_memory=cpu_memory,
            cpu_percent=cpu_percent,
            timestamp=time.time()
        )
        
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        return self._get_memory_stats()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        if not self.profile_data:
            return {}
            
        gpu_allocated = [s.gpu_allocated for s in self.profile_data]
        gpu_reserved = [s.gpu_reserved for s in self.profile_data]
        cpu_memory = [s.cpu_memory for s in self.profile_data]
        cpu_percent = [s.cpu_percent for s in self.profile_data]
        
        return {
            'gpu_allocated': {
                'min': min(gpu_allocated),
                'max': max(gpu_allocated),
                'avg': np.mean(gpu_allocated),
                'std': np.std(gpu_allocated)
            },
            'gpu_reserved': {
                'min': min(gpu_reserved),
                'max': max(gpu_reserved),
                'avg': np.mean(gpu_reserved),
                'std': np.std(gpu_reserved)
            },
            'cpu_memory': {
                'min': min(cpu_memory),
                'max': max(cpu_memory),
                'avg': np.mean(cpu_memory),
                'std': np.std(cpu_memory)
            },
            'cpu_percent': {
                'min': min(cpu_percent),
                'max': max(cpu_percent),
                'avg': np.mean(cpu_percent),
                'std': np.std(cpu_percent)
            },
            'duration': self.profile_data[-1].timestamp - self.profile_data[0].timestamp,
            'samples': len(self.profile_data)
        }
        
    def save_profile(self):
        """Save profile data to file"""
        if not self.save_path or not self.profile_data:
            return
            
        # Convert MemoryStats objects to dictionaries
        profile_dict = {
            'summary': self.get_summary(),
            'samples': [stat.to_dict() for stat in self.profile_data]
        }
        
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(profile_dict, f, indent=2)


class MemoryOptimizer:
    """Advanced memory optimization utilities"""
    
    def __init__(self, memory_threshold: float = 0.8, gc_threshold: float = 0.9):
        self.memory_threshold = memory_threshold
        self.gc_threshold = gc_threshold
        self.logger = logging.getLogger(__name__)
        
    def optimize_memory(self, model: nn.Module = None) -> Dict[str, Any]:
        """Optimize memory usage"""
        initial_stats = self._get_memory_stats()
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Model-specific optimizations
        if model is not None:
            self._optimize_model_memory(model)
            
        final_stats = self._get_memory_stats()
        
        return {
            'initial_memory': initial_stats,
            'final_memory': final_stats,
            'memory_freed': initial_stats['cpu_memory'] - final_stats['cpu_memory']
        }
        
    def _optimize_model_memory(self, model: nn.Module):
        """Apply model-specific memory optimizations"""
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        # Set model to eval mode to free some memory
        model.eval()
        
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        # GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_allocated = gpu_reserved = 0.0
            
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3
        cpu_percent = process.memory_percent()
        
        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved,
            'cpu_memory': cpu_memory,
            'cpu_percent': cpu_percent
        }
        
    def should_optimize(self) -> bool:
        """Check if memory optimization should be triggered"""
        stats = self._get_memory_stats()
        return stats['cpu_percent'] > self.memory_threshold * 100
        
    def should_gc(self) -> bool:
        """Check if garbage collection should be triggered"""
        stats = self._get_memory_stats()
        return stats['cpu_percent'] > self.gc_threshold * 100


@contextmanager
def memory_profiling(profile_interval: float = 1.0, save_path: Optional[str] = None):
    """Context manager for memory profiling"""
    profiler = MemoryProfiler(profile_interval, save_path)
    profiler.start_profiling()
    try:
        yield profiler
    finally:
        profiler.stop_profiling()


class AdaptiveBatchSizer:
    """Adaptive batch sizing based on memory usage"""
    
    def __init__(self, initial_batch_size: int = 8, min_batch_size: int = 1, 
                 max_batch_size: int = 64, memory_threshold: float = 0.8,
                 adaptation_factor: float = 0.1):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adaptation_factor = adaptation_factor
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.batch_history = []
        self.memory_history = []
        
    def get_batch_size(self) -> int:
        """Get current batch size"""
        return self.current_batch_size
        
    def update_batch_size(self, memory_usage: float, processing_time: float = None):
        """Update batch size based on memory usage and processing time"""
        self.memory_history.append(memory_usage)
        self.batch_history.append(self.current_batch_size)
        
        if memory_usage > self.memory_threshold:
            # Reduce batch size
            new_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adaptation_factor))
            )
            if new_size != self.current_batch_size:
                self.logger.info(f"Reducing batch size from {self.current_batch_size} to {new_size}")
                self.current_batch_size = new_size
        else:
            # Increase batch size gradually
            new_size = min(
                self.max_batch_size,
                int(self.current_batch_size * (1 + self.adaptation_factor))
            )
            if new_size != self.current_batch_size:
                self.logger.info(f"Increasing batch size from {self.current_batch_size} to {new_size}")
                self.current_batch_size = new_size
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch sizing statistics"""
        if not self.batch_history:
            return {}
            
        return {
            'current_batch_size': self.current_batch_size,
            'min_batch_size': min(self.batch_history),
            'max_batch_size': max(self.batch_history),
            'avg_batch_size': np.mean(self.batch_history),
            'avg_memory_usage': np.mean(self.memory_history),
            'max_memory_usage': max(self.memory_history),
            'adaptations': len(self.batch_history)
        }


# Enhanced memory utilities
def get_memory_summary() -> Dict[str, Any]:
    """Get comprehensive memory summary"""
    # GPU memory
    if torch.cuda.is_available():
        gpu_summary = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_memory_allocated': torch.cuda.max_memory_allocated() / 1024**3,
            'max_memory_reserved': torch.cuda.max_memory_reserved() / 1024**3,
        }
    else:
        gpu_summary = {'available': False}
    
    # CPU memory
    process = psutil.Process()
    cpu_summary = {
        'memory_gb': process.memory_info().rss / 1024**3,
        'memory_percent': process.memory_percent(),
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=0.1)
    }
    
    # System memory
    system_memory = psutil.virtual_memory()
    system_summary = {
        'total_gb': system_memory.total / 1024**3,
        'available_gb': system_memory.available / 1024**3,
        'used_gb': system_memory.used / 1024**3,
        'percent': system_memory.percent
    }
    
    return {
        'gpu': gpu_summary,
        'cpu': cpu_summary,
        'system': system_summary,
        'timestamp': time.time()
    }


def enable_memory_optimizations(model: nn.Module, enable_checkpointing: bool = True,
                               enable_mixed_precision: bool = True) -> Dict[str, bool]:
    """Enable various memory optimizations"""
    optimizations = {}
    
    # Gradient checkpointing
    if enable_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        optimizations['gradient_checkpointing'] = True
    else:
        optimizations['gradient_checkpointing'] = False
        
    # Mixed precision (requires external setup)
    optimizations['mixed_precision'] = enable_mixed_precision
    
    return optimizations


def cpu_offload_optimizer(optimizer, offload: bool = True):
    """CPU offload for optimizer states"""
    if not offload:
        return
        
    # This is a placeholder for CPU offload implementation
    # In practice, this would involve moving optimizer states to CPU
    # and managing the transfers during training
    pass


def activation_checkpointing(model: nn.Module, enable: bool = True):
    """Enable activation checkpointing"""
    if enable and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        return True
    return False


def clear_memory_cache():
    """Clear all memory caches"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
