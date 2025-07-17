# 🚨 DISCLAIMER 🚨

This repository is an **enhanced, extended, and modified version** of the amazing [RL2 project](https://github.com/ChenmienTan/RL2).

**All credit and super kudos go to Chenmien Tan** (the original author) and the RL2 team for their brilliant work and inspiration. This repo would not exist without their foundational code and ideas. If you use this, please check out and support the original RL2 project and Chenmien Tan—super cool guy!!!

# RL2: Ray Less Reinforcement Learning

A concise library of reinforcement learning for large language models.

This is the right library for you if you want to learn reinforcement learning for large language models or have a quick test for your own algorithm.
We deliver a clear implementation within 1K lines.


Despite the simplicity, you should be able to scale up to moderate-sized, *e.g.*, 72B, language models with

* Model partition via [Fully Sharded Data Parallelism](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) and [Tensor Parallelism](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
* Efficient sequence parallelism via [ZigZag Ring Attention](https://github.com/zhuzilin/ring-flash-attention)
* Inference engine and KV cache partition via Tensor Parallelism

We also support

* Balanced sequence packing for higher throughput
* Multi-turn rollout with [SGLang](https://github.com/sgl-project/sglang) async inference engine

RL2 is a production-ready library! Check our wandb report on [OpenThoughts](https://wandb.ai/chenmientan/OpenThoughts_archive), [SkyworkRM](https://wandb.ai/chenmientan/SkyworkRM_archive), [UltraFeedback](https://wandb.ai/chenmientan/UltraFeedback_archive), [OpenReasonerZero](https://wandb.ai/chenmientan/OpenReasonerZero_archive), and [SearchR1](https://wandb.ai/chenmientan/SearchR1_archive).

## Getting Started


### Installation

```
git clone https://github.com/ch33nchan/rl2.0.1.git
cd rl2.0.1
pip install -e .
```


### Data

Hugging Face dataset and various file types, *i.e.*, JSON, JSONL, CSV, Parquet, and Arrow, are accepted.
The data for SFT should be in the following format
```
[
    {
        "messages": [
            {"role": "user", "content": "What is the capital of China?"},
            {"role": "assistant", "content": "Beijing."}
        ]
    }
]
```
For RM and DPO
```
[
    {
        "messages": [
            {"role": "user", "content": "What is the capital of China?"}
        ],
        "chosen": "Beijing.",
        "rejected": "Shanghai."
    }
]
```
For PPO
```
[
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of China?"}
        ],
        "answer": "Beijing"
    }
]
```

For SFT, RM, and DPO, `batch_size` samples will be used for an update.
For PPO, `prompts_per_rollout` prompts will be used per rollout and `responses_per_prompt` trajectories will be sampled per prompt.
These trajectories will be evenly used for `update_per_rollout` updates.

### Rewards

The reward function should be in the follwing format.
Specify the path to the Python script including the function via `actor.rollout.env_path`.

```
def reward_fn(messages, answer):
    pred = parse_answer(messages[-1]["content"])
    return float(is_equivalent(pred, answer))
```

If a reward model is used, it should be served outside of the training framework, *e.g.*, using vLLM or SGLang, and be accessed in the reward function.

### Tools

RL2 supports multi-turn rollout with function calling.
In this case, you should set `rollout.max_turns > 1` and include function `interact` with the following format in the Python script including the reward function.
You should parse the called functions in past messages and return new messages including the results.
An empty list indicates no function is called.

```
def interact(messages):
    queries = parse_query(messages[-1]["content])
    results = [search(query) for query in queries]
    return [
        {"role": "tool", "content": result}
        for result in results
    ]
```
For base models, you may specify `rollout.apply_chat_template=false` so that the content in messages will be simply concatenated without applying chat template.

### Training

Use `torchrun` to launch the training. For example, for single node
```
torchrun \
    --nproc_per_node=<number of GPUs> \
    -m RL2.trainer.ppo \
    <args>
```
For multi nodes
```
torchrun \
    --nnodes=<number of nodes> \
    --node_rank=<rank of node> \
    --nproc_per_node=<number of GPUs on a node> \
    --master_addr=<address of master node> \
    --master_port=<port of master node> \
    -m RL2.trainer.ppo \
    <args>
```

## Guide for Hyper-Parameters

### Model Partition

* By default, *i.e.*, `ddp_size=1, tp_size=1`, your model will be partitioned via ZeRO stage 3.
* `ddp_size` specifies the number of model parameter copies.
For example, if you set `ddp_size` to the number of GPUs, your model will be partitioned by ZeRO stage 2.
Larger `ddp_size` leads to higher memory consumption and lower communication cost.
* For large models, sole data parallelism can be memory consuming.
You may specify `tp_size > 1` to enable tensor parallelism for higher throughput. 


### Sequence Length

For SFT, RM, and DPO, `max_length` is used to truncate sequences.
Notice that in RM and DPO, the chosen and rejected sequences will be packed together, so the actual sequence length can be up to twice of `max_length`.
For PPO, `max_new_tokens` is used to truncate generations.
The length of any sequence cannot exceed `sp_size * tp_size * max_length_per_device`.

### Algorithm

The default RL algorithm is [Dr. GRPO](https://arxiv.org/abs/2503.20783).
Specify `adv.estimator=gae` to use PPO or `adv.norm_var=true` and `kl.reward_estimator=k3` to use GRPO.

## Acknowledgement

This project is built upon the basis of many remarkable projects, including but not limited to
* [DeepSpeedChat](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) for the proposal of hybrid engine
* [RingFlashAttention](https://github.com/zhuzilin/ring-flash-attention) for the support of ZigZag ring attention
* [SGLang](https://github.com/sgl-project/sglang) for the support of async inference engine

We also thank [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [veRL](https://github.com/volcengine/verl) for their pioneering work.

## Improvements

---

## Extended RL2: Enhanced Features & Implementation Summary

> **Note:** This project is an **extended version** of the original [RL2 repository](https://github.com/ChenmienTan/RL2). All credit for the foundational code, design, and core ideas goes to the RL2 authors. This extension builds on their work, adding advanced features, improved memory optimization, multi-objective optimization, automated hyperparameter tuning, and modern MLOps integrations. The original RL2 repo is the main inspiration and reference for this project.

### Overview

This section documents the comprehensive enhancements made to RL2, including adaptive KL penalty mechanisms, multi-objective optimization, advanced advantage estimation, automated hyperparameter tuning, memory optimization, and experiment tracking. All features are implemented, tested, and documented.

### Key Improvements

- **Adaptive KL Penalty Mechanisms**: Exponential, linear, PID, and schedule-based controllers for stable policy optimization.
- **Constraint Optimization**: KL/entropy constraints, Lagrangian multipliers, and penalty-based constraint handling.
- **Multi-Objective Optimization**: Weighted sum, Tchebycheff, Pareto dominance, and hypervolume methods with Pareto frontier tracking.
- **Alternative Advantage Estimation**: GAE, V-trace, Retrace(λ), TD(λ), clipped IS, multi-step returns, and a unified interface.
- **Hyperparameter Optimization**: Random, grid, and Bayesian optimization (Optuna, scikit-optimize, hyperopt, bayesian-optimization).
- **Advanced Memory Optimization**: Profiling, adaptive batch sizing, CPU offloading, gradient checkpointing, and memory statistics.
- **Experiment Tracking & MLOps**: MLflow and Weights & Biases (W&B) integration, model versioning, and system metrics logging.

All enhancements are fully backward compatible and can be enabled or disabled via configuration.

### Usage Example

```python
from RL2.algs import AdaptiveKLController, MultiObjectiveOptimizer
from RL2.memory_optimizer import MemoryOptimizer
from RL2.experiment_tracking import ExperimentTracker

# Adaptive KL penalty
kl_controller = AdaptiveKLController(controller_type='exponential', target_kl=0.01, initial_coef=0.2)

# Multi-objective optimization
mo_optimizer = MultiObjectiveOptimizer(objectives=['reward', 'kl_penalty', 'entropy'], method='weighted_sum')

# Memory optimization
memory_optimizer = MemoryOptimizer(memory_threshold=0.8, gc_threshold=0.9)

# Experiment tracking
tracker = ExperimentTracker(experiment_name="RL2_Enhanced_Training", enable_mlflow=True, enable_wandb=True)
```

For more details, see the [Improvements](#improvements) section below and the configuration examples in `RL2/trainer/config/ppo.yaml`.

---

## Improvements

### 1. Adaptive KL Penalty Mechanisms
**Motivation:** In RL for language models, controlling the divergence between the new and reference policies is crucial for stable learning. Fixed KL penalties can lead to instability or suboptimal learning. Adaptive mechanisms dynamically adjust the penalty to maintain a target KL, improving both stability and sample efficiency.

**Methodology & Approach:**
- **Exponential Controller:** Adjusts the KL penalty exponentially based on the deviation from the target KL. Rapidly increases or decreases the penalty to quickly correct large KL errors.
- **Linear Controller:** Changes the penalty linearly, providing smoother, more predictable adjustments.
- **PID Controller:** Uses proportional, integral, and derivative terms to finely tune the penalty, minimizing oscillations and overshooting.
- **Schedule Controller:** Follows a predefined schedule (cosine, linear) for the penalty, useful for curriculum learning or staged training.
- **Implementation:** All controllers are implemented in `RL2/algs.py` as the `AdaptiveKLController` class, with a unified interface for easy switching.

### 2. Constraint Optimization
**Motivation:** RL objectives often require balancing multiple constraints (e.g., keeping KL below a threshold, maintaining entropy for exploration). Explicit constraint handling ensures safe and effective policy updates.

**Methodology & Approach:**
- **KL/Entropy Constraints:** Enforces hard or soft limits on KL divergence and entropy, preventing policy collapse or excessive drift.
- **Lagrangian Multipliers:** Automatically tunes constraint penalties to satisfy constraints during training.
- **Penalty Methods:** Adds penalties to the loss when constraints are violated, guiding optimization back to feasible regions.
- **Implementation:** Provided via the `ConstraintOptimizer` class in `RL2/algs.py`, supporting both hard and soft constraint modes.

### 3. Multi-Objective Optimization
**Motivation:** Real-world RL often involves optimizing for multiple objectives (e.g., reward, safety, diversity). Simple reward shaping is insufficient for complex trade-offs.

**Methodology & Approach:**
- **Weighted Sum:** Combines objectives with user-defined weights for simple trade-offs.
- **Tchebycheff:** Focuses on minimizing the maximum deviation from ideal objectives, useful for fairness.
- **Pareto Dominance:** Maintains a set of non-dominated solutions, allowing users to select from the Pareto frontier.
- **Hypervolume:** Quantifies the quality of the Pareto set, enabling automated selection of diverse, high-quality solutions.
- **Implementation:** All methods are available in the `MultiObjectiveOptimizer` class in `RL2/algs.py`, with efficient Pareto archive management.

### 4. Alternative Advantage Estimation Methods
**Motivation:** The choice of advantage estimator impacts bias, variance, and sample efficiency. Supporting multiple estimators allows users to tailor RL2 to their problem and data regime.

**Methodology & Approach:**
- **GAE:** Balances bias and variance via λ parameter, standard for PPO.
- **V-trace & Retrace(λ):** Off-policy estimators with importance sampling and clipping for stability.
- **TD(λ):** Multi-step bootstrapped returns for improved learning.
- **Clipped IS & Multi-step:** Further reduce variance and improve off-policy robustness.
- **Unified API:** All estimators share a common interface, making it easy to switch or combine methods.
- **Implementation:** Functions are in `RL2/algs.py`, with clear documentation and tests.

### 5. Hyperparameter Optimization
**Motivation:** Manual hyperparameter tuning is time-consuming and suboptimal. Automated search accelerates development and improves results.

**Methodology & Approach:**
- **Random/Grid Search:** Baseline methods for quick or exhaustive exploration.
- **Bayesian Optimization:** Uses probabilistic models to efficiently explore high-dimensional spaces.
- **Multiple Backends:** Supports Optuna, scikit-optimize, Hyperopt, and bayesian-optimization for flexibility.
- **Result Tracking:** Stores all trials, best parameters, and optimization history for reproducibility.
- **Implementation:** The `HyperparameterTuner` class in `RL2/hyperopt.py` provides a unified interface and logging.

### 6. Advanced Memory Optimization
**Motivation:** Training large models is often bottlenecked by memory. Advanced memory management enables larger models and longer sequences on the same hardware.

**Methodology & Approach:**
- **Profiling:** Monitors memory usage in real time, identifying bottlenecks.
- **Adaptive Batch Sizing:** Dynamically adjusts batch size to maximize GPU utilization without OOM errors.
- **CPU Offloading:** Moves tensors to CPU when not needed on GPU, freeing up space.
- **Gradient Checkpointing:** Saves memory by recomputing activations during backward pass.
- **Garbage Collection:** Proactively frees unused memory to prevent leaks.
- **Implementation:** All features are in `RL2/memory_optimizer.py`, with user-configurable thresholds and reporting.

### 7. Experiment Tracking & MLOps Integration
**Motivation:** Reproducibility, collaboration, and model management are essential for modern ML workflows. Integrated tracking and versioning streamline research and deployment.

**Methodology & Approach:**
- **MLflow & W&B:** Log metrics, parameters, artifacts, and system stats in real time.
- **Model Versioning:** Automatically saves and versions models for easy rollback and comparison.
- **System Metrics:** Tracks hardware utilization for performance analysis.
- **Implementation:** The `ExperimentTracker` class in `RL2/experiment_tracking.py` abstracts both MLflow and W&B, with simple enable/disable toggles.

---

## Design Philosophy & Extensibility

- **Modular Architecture:** All enhancements are implemented as modular, pluggable components. Users can enable, disable, or extend any feature via configuration or subclassing.
- **Backward Compatibility:** All new features are opt-in; existing RL2 workflows remain unchanged unless enhancements are explicitly enabled.
- **Extensible API:** New optimizers, estimators, or tracking backends can be added with minimal code changes, following the provided extension points.
- **Comprehensive Documentation:** Every feature is documented in `ENHANCED_FEATURES.md` and the codebase, with usage examples and best practices.

---
