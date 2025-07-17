import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def differentiable_all_reduce(tensor, device_mesh):

    detached_tensor = tensor.detach()
    dist.all_reduce(
        detached_tensor,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    return tensor + detached_tensor - tensor.detach()

def sequence_all_reduce(tensor, cu_seqlens, device_mesh):

    tensor = torch.stack([
        tensor[:, start_idx:end_idx].sum()
        for start_idx, end_idx
        in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ])
    return differentiable_all_reduce(tensor, device_mesh)

def compute_logsumexp(logits, device_mesh, chunk_size=1024):

    # When using tensor parallelism, each device only has a shard of logits.
    # We firstly compute logsumexp of the sharded logits on each device,
    # and then perform logsumexp across devices, which is equivalent to 
    # performing logsumexp over the entire vocabulary.

    # Direct logsumexp over the entire sequence suffer high memory peak.
    # See https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    logsumexps = []
    for start in range(0, logits.shape[1], chunk_size):
        logsumexp = torch.logsumexp(
            logits[:, start:start + chunk_size], -1
        )
        logsumexps.append(logsumexp)
    logsumexp = torch.cat(logsumexps, -1)

    logsumexps = [
        torch.zeros_like(logsumexp)
        for _ in range(device_mesh.size())
    ]
    dist.all_gather(
        logsumexps,
        logsumexp,
        group=device_mesh.get_group()
    )
    logsumexps[device_mesh.get_local_rank()] = logsumexp # necessary to retain grad
    logsumexps = torch.cat([
        logsumexp.unsqueeze(-1) for logsumexp in logsumexps
    ], -1)
    return torch.logsumexp(logsumexps, -1)

def gather_action_logits(logits, actions, device_mesh):

    # When using tensor parallelism, each device only has a shard of logits.
    # On each device, we gather logits for actions on the device, and then 
    # perform AllReduce to collect the complete logits.
    rank = device_mesh.get_local_rank()

    local_vocab_size = torch.LongTensor(
        [logits.shape[-1]]
    ).to(torch.cuda.current_device())
    vocab_sizes = [
        torch.zeros_like(local_vocab_size)
        for _ in range(device_mesh.size())
    ]
    dist.all_gather(
        vocab_sizes,
        local_vocab_size,
        group=device_mesh.get_group()
    )
    cu_vocab_sizes = torch.cumsum(
        torch.cat(
            [torch.zeros_like(local_vocab_size)] + vocab_sizes
        ), 0
    )
    action_device_mapping = (
        actions < cu_vocab_sizes[1:].unsqueeze(-1)
    ).to(torch.float32).argmax(0)
    local_action_indices = torch.where(
        action_device_mapping == rank
    )[0]
    local_actions = actions[:, local_action_indices] - cu_vocab_sizes[rank]
    action_logits = torch.zeros(
        actions.shape, device=torch.cuda.current_device()
    )
    action_logits[:, local_action_indices] = torch.gather(
        logits[:, local_action_indices],
        dim=-1,
        index=local_actions.unsqueeze(-1)
    ).squeeze(-1)

    return differentiable_all_reduce(action_logits, device_mesh)

def compute_entropy(logits, logsumexp, device_mesh):

    probs = torch.exp(logits - logsumexp.unsqueeze(-1))
    return logsumexp - differentiable_all_reduce(
        (probs * logits).sum(-1), device_mesh
    )

def compute_approx_kl(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    estimator: str
) -> torch.Tensor:
    # The (ref_)logps of non-action tokens are zero (see `Actor.
    # forward`), so their corresponding kl_term will also be zero.

    log_ratio = logps - ref_logps
    if estimator == "k1":
        return log_ratio
    elif estimator == "k2":
        return log_ratio.pow(2) / 2
    elif estimator == "k3":
        return log_ratio + torch.exp(- log_ratio) - 1
    else:
        raise NotImplementedError

def compute_gae(data_list, gamma, lamda):
    """
    Compute Generalized Advantage Estimation (GAE)
    """
    # extract rewards and values of action tokens
    rewards, values, action_mask = [], [], []
    for ex in data_list:
        indices = torch.where(ex["action_mask"])[0]
        rewards.append(ex["rewards"][indices])
        values.append(ex["values"][indices])
        action_mask.append(ex["action_mask"][indices])
    # pad to identical length for efficient computation
    rewards = pad_sequence(rewards, True)
    values = pad_sequence(values, True)
    action_mask = pad_sequence(action_mask, True)
    
    # \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
    next_values = torch.cat((values[:, 1:], torch.zeros((values.shape[0], 1))), -1)
    deltas = rewards + gamma * next_values - values

    # A_t = \delta_t + \gamma * \lambda * A_{t+1}
    gae, reversed_gaes = 0, []
    for t in reversed(range(deltas.shape[-1])):
        gae = deltas[:, t] + gamma * lamda * gae
        reversed_gaes.append(gae)
    gaes = torch.stack(reversed_gaes[::-1], -1)
    returns = gaes + values

    action_gaes = gaes[torch.where(action_mask)]
    gaes = (gaes - action_gaes.mean()) * action_mask / (
        action_gaes.std() + torch.finfo(gaes.dtype).eps
    )

    for ex, gae, ret in zip(data_list, gaes, returns):
        ex["advantages"] = torch.zeros_like(ex["rewards"])
        ex["returns"] = torch.zeros_like(ex["rewards"])
        indices = torch.where(ex["action_mask"])[0]
        ex["advantages"][indices] = gae[:len(indices)]
        ex["returns"][indices] = ret[:len(indices)]

def compute_reinforce_adv(data_list, responses_per_prompt, norm_var: bool):
    """
    Compute REINFORCE advantage estimation
    """
    rewards = torch.FloatTensor(
        [ex["rewards"].sum() for ex in data_list]
    ).view(-1, responses_per_prompt)
    baseline = rewards.mean(-1)
    advantages = rewards - baseline.unsqueeze(-1)

    if norm_var:
        stds = rewards.std(-1)
        advantages /= (
            stds.unsqueeze(-1) + torch.finfo(advantages.dtype).eps
        )

    for ex, advantage in zip(data_list, advantages.flatten()):
        ex["advantages"] = advantage * ex["action_mask"]

def fill_zero_adv(data_list):
    """
    Fill advantages with zeros
    """
    for ex in data_list:
        ex["advantages"] = torch.zeros_like(ex["rewards"])

class AdaptiveKLController:
    """
    Adaptive KL penalty controller with multiple strategies:
    - Exponential: Exponential moving average based adaptation
    - Linear: Linear increase/decrease based on KL divergence
    - PID: Proportional-Integral-Derivative controller
    - Schedule: Pre-defined schedule based controller
    """
    
    def __init__(self, 
                 initial_coef: float = 0.2,
                 target_kl: float = 0.01,
                 controller_type: str = "exponential",
                 # Exponential controller params
                 exp_factor: float = 1.03,
                 exp_decay: float = 0.99,
                 # Linear controller params
                 linear_factor: float = 0.5,
                 # PID controller params
                 kp: float = 0.1,
                 ki: float = 0.01,
                 kd: float = 0.001,
                 # Schedule controller params
                 schedule_type: str = "cosine",
                 schedule_steps: int = 1000,
                 # General bounds
                 min_coef: float = 0.001,
                 max_coef: float = 10.0):
        
        self.coef = initial_coef
        self.target_kl = target_kl
        self.controller_type = controller_type
        self.min_coef = min_coef
        self.max_coef = max_coef
        
        # Exponential controller
        self.exp_factor = exp_factor
        self.exp_decay = exp_decay
        
        # Linear controller
        self.linear_factor = linear_factor
        
        # PID controller
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.previous_error = 0.0
        
        # Schedule controller
        self.schedule_type = schedule_type
        self.schedule_steps = schedule_steps
        self.initial_coef = initial_coef
        self.step_count = 0
        
        # History for analysis
        self.kl_history = []
        self.coef_history = []
    
    def update(self, current_kl: float, step: int = None) -> float:
        """Update KL coefficient based on current KL divergence"""
        self.kl_history.append(current_kl)
        self.step_count += 1
        
        if self.controller_type == "exponential":
            self.coef = self._exponential_update(current_kl)
        elif self.controller_type == "linear":
            self.coef = self._linear_update(current_kl)
        elif self.controller_type == "pid":
            self.coef = self._pid_update(current_kl)
        elif self.controller_type == "schedule":
            self.coef = self._schedule_update(step or self.step_count)
        elif self.controller_type == "fixed":
            pass  # Keep coefficient fixed
        else:
            raise ValueError(f"Unknown controller type: {self.controller_type}")
        
        # Apply bounds
        self.coef = max(self.min_coef, min(self.max_coef, self.coef))
        self.coef_history.append(self.coef)
        
        return self.coef
    
    def _exponential_update(self, current_kl: float) -> float:
        """Exponential moving average based adaptation"""
        if current_kl > self.target_kl:
            return self.coef * self.exp_factor
        else:
            return self.coef * self.exp_decay
    
    def _linear_update(self, current_kl: float) -> float:
        """Linear adaptation based on KL divergence"""
        kl_ratio = current_kl / self.target_kl
        adjustment = self.linear_factor * (kl_ratio - 1.0)
        return self.coef * (1.0 + adjustment)
    
    def _pid_update(self, current_kl: float) -> float:
        """PID controller for KL coefficient"""
        error = current_kl - self.target_kl
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.previous_error)
        self.previous_error = error
        
        # PID output
        pid_output = p_term + i_term + d_term
        return max(0.0, self.coef + pid_output)
    
    def _schedule_update(self, step: int) -> float:
        """Schedule-based coefficient updates"""
        progress = min(step / self.schedule_steps, 1.0)
        
        if self.schedule_type == "cosine":
            coef = self.initial_coef * (1 + np.cos(np.pi * progress)) / 2
        elif self.schedule_type == "linear":
            coef = self.initial_coef * (1 - progress)
        elif self.schedule_type == "exponential":
            coef = self.initial_coef * np.exp(-progress * 3)
        elif self.schedule_type == "warmup_cosine":
            if progress < 0.1:  # Warmup phase
                coef = self.initial_coef * progress / 0.1
            else:
                adjusted_progress = (progress - 0.1) / 0.9
                coef = self.initial_coef * (1 + np.cos(np.pi * adjusted_progress)) / 2
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return coef
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.step_count = 0
        self.kl_history = []
        self.coef_history = []
    
    def get_stats(self) -> dict:
        """Get controller statistics"""
        return {
            "current_coef": self.coef,
            "target_kl": self.target_kl,
            "current_kl": self.kl_history[-1] if self.kl_history else 0.0,
            "avg_kl": np.mean(self.kl_history) if self.kl_history else 0.0,
            "controller_type": self.controller_type,
            "step_count": self.step_count
        }

class ConstraintOptimizer:
    """
    Constraint optimization for PPO with various constraint types:
    - KL constraint: Hard constraint on KL divergence
    - Entropy constraint: Maintain minimum entropy
    - Gradient constraint: Clip gradients based on constraint violations
    """
    
    def __init__(self,
                 kl_constraint: float = 0.01,
                 entropy_constraint: float = 0.01,
                 use_lagrangian: bool = True,
                 lagrangian_lr: float = 0.01,
                 constraint_violation_penalty: float = 10.0):
        
        self.kl_constraint = kl_constraint
        self.entropy_constraint = entropy_constraint
        self.use_lagrangian = use_lagrangian
        self.lagrangian_lr = lagrangian_lr
        self.constraint_violation_penalty = constraint_violation_penalty
        
        # Lagrangian multipliers
        self.kl_multiplier = 1.0
        self.entropy_multiplier = 1.0
        
        # Constraint violation history
        self.kl_violations = []
        self.entropy_violations = []
    
    def check_constraints(self, kl_div: float, entropy: float) -> dict:
        """Check if constraints are satisfied"""
        kl_violation = max(0, kl_div - self.kl_constraint)
        entropy_violation = max(0, self.entropy_constraint - entropy)
        
        self.kl_violations.append(kl_violation)
        self.entropy_violations.append(entropy_violation)
        
        return {
            "kl_satisfied": kl_violation == 0,
            "entropy_satisfied": entropy_violation == 0,
            "kl_violation": kl_violation,
            "entropy_violation": entropy_violation
        }
    
    def compute_constraint_loss(self, kl_div: float, entropy: float) -> torch.Tensor:
        """Compute constraint-based loss"""
        constraint_loss = torch.tensor(0.0, requires_grad=True)
        
        # KL constraint loss
        if kl_div > self.kl_constraint:
            kl_penalty = self.constraint_violation_penalty * (kl_div - self.kl_constraint) ** 2
            constraint_loss = constraint_loss + kl_penalty
        
        # Entropy constraint loss
        if entropy < self.entropy_constraint:
            entropy_penalty = self.constraint_violation_penalty * (self.entropy_constraint - entropy) ** 2
            constraint_loss = constraint_loss + entropy_penalty
        
        return constraint_loss
    
    def update_lagrangian_multipliers(self, kl_div: float, entropy: float):
        """Update Lagrangian multipliers based on constraint violations"""
        if not self.use_lagrangian:
            return
        
        # Update KL multiplier
        kl_violation = kl_div - self.kl_constraint
        self.kl_multiplier = max(0, self.kl_multiplier + self.lagrangian_lr * kl_violation)
        
        # Update entropy multiplier
        entropy_violation = self.entropy_constraint - entropy
        self.entropy_multiplier = max(0, self.entropy_multiplier + self.lagrangian_lr * entropy_violation)
    
    def get_lagrangian_loss(self, kl_div: float, entropy: float) -> torch.Tensor:
        """Compute Lagrangian-based constraint loss"""
        if not self.use_lagrangian:
            return torch.tensor(0.0)
        
        kl_term = self.kl_multiplier * max(0, kl_div - self.kl_constraint)
        entropy_term = self.entropy_multiplier * max(0, self.entropy_constraint - entropy)
        
        return kl_term + entropy_term
    
    def get_stats(self) -> dict:
        """Get constraint optimizer statistics"""
        return {
            "kl_constraint": self.kl_constraint,
            "entropy_constraint": self.entropy_constraint,
            "kl_multiplier": self.kl_multiplier,
            "entropy_multiplier": self.entropy_multiplier,
            "avg_kl_violation": np.mean(self.kl_violations) if self.kl_violations else 0.0,
            "avg_entropy_violation": np.mean(self.entropy_violations) if self.entropy_violations else 0.0,
            "constraint_violations": len([v for v in self.kl_violations if v > 0]) + len([v for v in self.entropy_violations if v > 0])
        }

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for PPO with Pareto frontier computation.
    Supports multiple objectives like reward, KL divergence, entropy, etc.
    """
    
    def __init__(self,
                 objectives: list = ["reward", "kl_penalty", "entropy"],
                 objective_weights: dict = None,
                 pareto_method: str = "weighted_sum",
                 diversity_weight: float = 0.1,
                 archive_size: int = 100):
        
        self.objectives = objectives
        self.objective_weights = objective_weights or {obj: 1.0 for obj in objectives}
        self.pareto_method = pareto_method
        self.diversity_weight = diversity_weight
        self.archive_size = archive_size
        
        # Pareto archive for storing non-dominated solutions
        self.pareto_archive = []
        self.objective_history = []
        
        # Reference point for hypervolume calculation
        self.reference_point = None
        
    def compute_pareto_loss(self, objectives_dict: dict) -> torch.Tensor:
        """Compute loss based on multi-objective optimization strategy"""
        
        if self.pareto_method == "weighted_sum":
            return self._weighted_sum_loss(objectives_dict)
        elif self.pareto_method == "tchebycheff":
            return self._tchebycheff_loss(objectives_dict)
        elif self.pareto_method == "pareto_dominance":
            return self._pareto_dominance_loss(objectives_dict)
        elif self.pareto_method == "hypervolume":
            return self._hypervolume_loss(objectives_dict)
        else:
            raise ValueError(f"Unknown Pareto method: {self.pareto_method}")
    
    def _weighted_sum_loss(self, objectives_dict: dict) -> torch.Tensor:
        """Weighted sum scalarization"""
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        for obj_name, obj_value in objectives_dict.items():
            if obj_name in self.objective_weights:
                weight = self.objective_weights[obj_name]
                total_loss = total_loss + weight * obj_value
        
        return total_loss
    
    def _tchebycheff_loss(self, objectives_dict: dict) -> torch.Tensor:
        """Tchebycheff scalarization for better Pareto frontier coverage"""
        if not self.pareto_archive:
            return self._weighted_sum_loss(objectives_dict)
        
        # Use ideal point from archive
        ideal_point = self._compute_ideal_point()
        
        max_weighted_deviation = torch.tensor(0.0, requires_grad=True)
        for obj_name, obj_value in objectives_dict.items():
            if obj_name in self.objective_weights:
                weight = self.objective_weights[obj_name]
                ideal_value = ideal_point.get(obj_name, 0.0)
                deviation = weight * abs(obj_value - ideal_value)
                max_weighted_deviation = torch.max(max_weighted_deviation, deviation)
        
        return max_weighted_deviation
    
    def _pareto_dominance_loss(self, objectives_dict: dict) -> torch.Tensor:
        """Loss based on Pareto dominance ranking"""
        if not self.pareto_archive:
            return self._weighted_sum_loss(objectives_dict)
        
        # Compute dominance rank
        dominance_rank = self._compute_dominance_rank(objectives_dict)
        
        # Convert rank to loss (higher rank = higher loss)
        base_loss = self._weighted_sum_loss(objectives_dict)
        dominance_penalty = torch.tensor(dominance_rank * 0.1, requires_grad=True)
        
        return base_loss + dominance_penalty
    
    def _hypervolume_loss(self, objectives_dict: dict) -> torch.Tensor:
        """Loss based on hypervolume contribution"""
        if not self.pareto_archive or not self.reference_point:
            return self._weighted_sum_loss(objectives_dict)
        
        # Compute hypervolume contribution
        hv_contribution = self._compute_hypervolume_contribution(objectives_dict)
        
        # Convert to loss (negative hypervolume contribution)
        base_loss = self._weighted_sum_loss(objectives_dict)
        hv_bonus = torch.tensor(-hv_contribution * self.diversity_weight, requires_grad=True)
        
        return base_loss + hv_bonus
    
    def update_pareto_archive(self, objectives_dict: dict, solution_data: dict = None):
        """Update Pareto archive with new solution"""
        new_solution = {
            "objectives": objectives_dict.copy(),
            "data": solution_data or {},
            "timestamp": len(self.objective_history)
        }
        
        # Check if new solution is dominated
        dominated = False
        to_remove = []
        
        for i, existing_solution in enumerate(self.pareto_archive):
            if self._dominates(existing_solution["objectives"], objectives_dict):
                dominated = True
                break
            elif self._dominates(objectives_dict, existing_solution["objectives"]):
                to_remove.append(i)
        
        # Remove dominated solutions
        for i in reversed(to_remove):
            self.pareto_archive.pop(i)
        
        # Add new solution if not dominated
        if not dominated:
            self.pareto_archive.append(new_solution)
        
        # Maintain archive size
        if len(self.pareto_archive) > self.archive_size:
            self._prune_archive()
        
        # Update history
        self.objective_history.append(objectives_dict.copy())
    
    def _dominates(self, obj1: dict, obj2: dict) -> bool:
        """Check if obj1 dominates obj2 (assuming minimization)"""
        better_in_one = False
        
        for obj_name in self.objectives:
            if obj_name in obj1 and obj_name in obj2:
                if obj1[obj_name] > obj2[obj_name]:  # obj1 is worse
                    return False
                elif obj1[obj_name] < obj2[obj_name]:  # obj1 is better
                    better_in_one = True
        
        return better_in_one
    
    def _compute_ideal_point(self) -> dict:
        """Compute ideal point from Pareto archive"""
        if not self.pareto_archive:
            return {}
        
        ideal_point = {}
        for obj_name in self.objectives:
            values = [sol["objectives"].get(obj_name, float('inf')) for sol in self.pareto_archive]
            ideal_point[obj_name] = min(values)
        
        return ideal_point
    
    def _compute_dominance_rank(self, objectives_dict: dict) -> int:
        """Compute dominance rank of current solution"""
        rank = 0
        for solution in self.pareto_archive:
            if self._dominates(solution["objectives"], objectives_dict):
                rank += 1
        return rank
    
    def _compute_hypervolume_contribution(self, objectives_dict: dict) -> float:
        """Compute hypervolume contribution (simplified implementation)"""
        if not self.reference_point:
            return 0.0
        
        # Simplified hypervolume calculation
        contribution = 1.0
        for obj_name in self.objectives:
            if obj_name in objectives_dict and obj_name in self.reference_point:
                obj_value = objectives_dict[obj_name]
                ref_value = self.reference_point[obj_name]
                contribution *= max(0, ref_value - obj_value)
        
        return contribution
    
    def _prune_archive(self):
        """Prune archive to maintain size limit using diversity-based selection"""
        if len(self.pareto_archive) <= self.archive_size:
            return
        
        # Sort by diversity (simplified: use timestamp)
        self.pareto_archive.sort(key=lambda x: x["timestamp"], reverse=True)
        self.pareto_archive = self.pareto_archive[:self.archive_size]
    
    def get_pareto_front(self) -> list:
        """Get current Pareto front"""
        return [sol["objectives"] for sol in self.pareto_archive]
    
    def get_stats(self) -> dict:
        """Get multi-objective optimizer statistics"""
        return {
            "pareto_size": len(self.pareto_archive),
            "total_evaluations": len(self.objective_history),
            "pareto_method": self.pareto_method,
            "ideal_point": self._compute_ideal_point(),
            "objective_ranges": self._compute_objective_ranges()
        }
    
    def _compute_objective_ranges(self) -> dict:
        """Compute objective value ranges from history"""
        if not self.objective_history:
            return {}
        
        ranges = {}
        for obj_name in self.objectives:
            values = [obj_dict.get(obj_name, 0.0) for obj_dict in self.objective_history]
            ranges[obj_name] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "std": np.std(values)
            }
        
        return ranges

def compute_vtrace_adv(data_list, gamma: float = 0.99, rho_bar: float = 1.0, c_bar: float = 1.0):
    """
    V-trace advantage estimation for off-policy learning
    Args:
        data_list: List of trajectory data
        gamma: Discount factor
        rho_bar: Clipping threshold for importance sampling ratios
        c_bar: Clipping threshold for trace cutting
    """
    for ex in data_list:
        indices = torch.where(ex["action_mask"])[0]
        if len(indices) == 0:
            ex["advantages"] = torch.zeros_like(ex["rewards"])
            continue
        
        rewards = ex["rewards"][indices]
        values = ex["values"][indices]
        logps = ex["logps"][indices]
        ref_logps = ex["ref_logps"][indices]
        
        # Compute importance sampling ratios
        log_rhos = logps - ref_logps
        rhos = torch.exp(log_rhos)
        
        # Clip importance sampling ratios
        rhos_clipped = torch.clamp(rhos, max=rho_bar)
        cs = torch.clamp(rhos, max=c_bar)
        
        # Compute V-trace targets
        vtrace_targets = []
        vtrace_target = values[-1]  # Bootstrap from last value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                vtrace_target = rewards[t] + gamma * vtrace_target
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                vtrace_target = values[t] + rhos_clipped[t] * delta + gamma * cs[t] * (vtrace_target - values[t + 1])
            vtrace_targets.append(vtrace_target)
        
        vtrace_targets = torch.stack(vtrace_targets[::-1])
        advantages = vtrace_targets - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Fill back to original structure
        ex["advantages"] = torch.zeros_like(ex["rewards"])
        ex["advantages"][indices] = advantages

def compute_retrace_adv(data_list, gamma: float = 0.99, lamda: float = 0.95):
    """
    Retrace(λ) advantage estimation for safe off-policy learning
    Args:
        data_list: List of trajectory data
        gamma: Discount factor
        lamda: Trace decay parameter
    """
    for ex in data_list:
        indices = torch.where(ex["action_mask"])[0]
        if len(indices) == 0:
            ex["advantages"] = torch.zeros_like(ex["rewards"])
            continue
        
        rewards = ex["rewards"][indices]
        values = ex["values"][indices]
        logps = ex["logps"][indices]
        ref_logps = ex["ref_logps"][indices]
        
        # Compute importance sampling ratios
        log_rhos = logps - ref_logps
        rhos = torch.exp(log_rhos)
        
        # Compute Retrace targets
        retrace_targets = []
        retrace_target = values[-1]  # Bootstrap from last value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                retrace_target = rewards[t] + gamma * retrace_target
            else:
                c_t = lamda * torch.min(torch.tensor(1.0), rhos[t])
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                retrace_target = values[t] + c_t * delta + gamma * c_t * (retrace_target - values[t + 1])
            retrace_targets.append(retrace_target)
        
        retrace_targets = torch.stack(retrace_targets[::-1])
        advantages = retrace_targets - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Fill back to original structure
        ex["advantages"] = torch.zeros_like(ex["rewards"])
        ex["advantages"][indices] = advantages

def compute_td_lambda_adv(data_list, gamma: float = 0.99, lamda: float = 0.95):
    """
    TD(λ) advantage estimation
    Args:
        data_list: List of trajectory data
        gamma: Discount factor
        lamda: Eligibility trace decay
    """
    for ex in data_list:
        indices = torch.where(ex["action_mask"])[0]
        if len(indices) == 0:
            ex["advantages"] = torch.zeros_like(ex["rewards"])
            continue
        
        rewards = ex["rewards"][indices]
        values = ex["values"][indices]
        
        # Compute TD errors
        next_values = torch.cat([values[1:], torch.zeros(1)])
        td_errors = rewards + gamma * next_values - values
        
        # Compute TD(λ) returns using eligibility traces
        td_lambda_returns = []
        td_lambda_return = values[-1]  # Bootstrap from last value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                td_lambda_return = rewards[t] + gamma * td_lambda_return
            else:
                td_lambda_return = values[t] + td_errors[t] + gamma * lamda * (td_lambda_return - next_values[t])
            td_lambda_returns.append(td_lambda_return)
        
        td_lambda_returns = torch.stack(td_lambda_returns[::-1])
        advantages = td_lambda_returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Fill back to original structure
        ex["advantages"] = torch.zeros_like(ex["rewards"])
        ex["advantages"][indices] = advantages

def compute_clipped_is_adv(data_list, gamma: float = 0.99, clip_ratio: float = 0.2):
    """
    Clipped importance sampling advantage estimation
    Args:
        data_list: List of trajectory data
        gamma: Discount factor
        clip_ratio: Clipping ratio for importance sampling
    """
    for ex in data_list:
        indices = torch.where(ex["action_mask"])[0]
        if len(indices) == 0:
            ex["advantages"] = torch.zeros_like(ex["rewards"])
            continue
        
        rewards = ex["rewards"][indices]
        values = ex["values"][indices]
        logps = ex["logps"][indices]
        ref_logps = ex["ref_logps"][indices]
        
        # Compute importance sampling ratios
        log_rhos = logps - ref_logps
        rhos = torch.exp(log_rhos)
        
        # Clip importance sampling ratios
        rhos_clipped = torch.clamp(rhos, 1.0 - clip_ratio, 1.0 + clip_ratio)
        
        # Compute discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        
        for t in reversed(range(len(rewards))):
            discounted_reward = rewards[t] + gamma * discounted_reward
            discounted_rewards.append(discounted_reward)
        
        discounted_rewards = torch.stack(discounted_rewards[::-1])
        
        # Apply importance sampling correction
        is_corrected_returns = rhos_clipped * discounted_rewards
        advantages = is_corrected_returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Fill back to original structure
        ex["advantages"] = torch.zeros_like(ex["rewards"])
        ex["advantages"][indices] = advantages

def compute_multistep_adv(data_list, gamma: float = 0.99, n_steps: int = 5, bootstrap_method: str = "gae"):
    """
    Multi-step advantage estimation with different bootstrapping strategies
    Args:
        data_list: List of trajectory data
        gamma: Discount factor
        n_steps: Number of steps for multi-step returns
        bootstrap_method: Method for bootstrapping ('gae', 'td', 'mc')
    """
    for ex in data_list:
        indices = torch.where(ex["action_mask"])[0]
        if len(indices) == 0:
            ex["advantages"] = torch.zeros_like(ex["rewards"])
            continue
        
        rewards = ex["rewards"][indices]
        values = ex["values"][indices]
        
        # Compute multi-step returns
        multistep_returns = []
        
        for t in range(len(rewards)):
            multistep_return = 0
            
            # Sum up n-step rewards
            for k in range(min(n_steps, len(rewards) - t)):
                multistep_return += (gamma ** k) * rewards[t + k]
            
            # Bootstrap with different methods
            if t + n_steps < len(rewards):
                if bootstrap_method == "gae":
                    # Use GAE-style bootstrapping
                    bootstrap_value = values[t + n_steps]
                    multistep_return += (gamma ** n_steps) * bootstrap_value
                elif bootstrap_method == "td":
                    # Use TD-style bootstrapping
                    bootstrap_value = values[t + n_steps]
                    multistep_return += (gamma ** n_steps) * bootstrap_value
                elif bootstrap_method == "mc":
                    # Use Monte Carlo (no bootstrapping)
                    pass
            
            multistep_returns.append(multistep_return)
        
        multistep_returns = torch.stack(multistep_returns)
        advantages = multistep_returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Fill back to original structure
        ex["advantages"] = torch.zeros_like(ex["rewards"])
        ex["advantages"][indices] = advantages

def compute_advantages_unified(data_list, method: str = "gae", **kwargs):
    """
    Unified advantage computation function supporting multiple methods
    Args:
        data_list: List of trajectory data
        method: Advantage estimation method
        **kwargs: Method-specific parameters
    """
    if method == "gae":
        compute_gae(data_list, kwargs.get("gamma", 0.99), kwargs.get("lamda", 0.95))
    elif method == "reinforce":
        compute_reinforce_adv(data_list, kwargs.get("responses_per_prompt", 1), kwargs.get("norm_var", False))
    elif method == "vtrace":
        compute_vtrace_adv(data_list, kwargs.get("gamma", 0.99), kwargs.get("rho_bar", 1.0), kwargs.get("c_bar", 1.0))
    elif method == "retrace":
        compute_retrace_adv(data_list, kwargs.get("gamma", 0.99), kwargs.get("lamda", 0.95))
    elif method == "td_lambda":
        compute_td_lambda_adv(data_list, kwargs.get("gamma", 0.99), kwargs.get("lamda", 0.95))
    elif method == "clipped_is":
        compute_clipped_is_adv(data_list, kwargs.get("gamma", 0.99), kwargs.get("clip_ratio", 0.2))
    elif method == "multistep":
        compute_multistep_adv(data_list, kwargs.get("gamma", 0.99), kwargs.get("n_steps", 5), kwargs.get("bootstrap_method", "gae"))
    elif method == "zero":
        fill_zero_adv(data_list)
    else:
        raise ValueError(f"Unknown advantage estimation method: {method}")