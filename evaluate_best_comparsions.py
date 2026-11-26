import argparse
import csv
import json
import math
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.half_cheetah_combined_max_cost import (
    COMBINED_COST_VERSION,
    HalfCheetahCombinedMaxCost,
)
from envs.half_cheetah_jointspeed_cost import (
    COST_VERSION as JOINT_SPEED_COST_VERSION,
    HalfCheetahJointSpeedCost,
)
from envs.half_cheetah_multi_constraint_cost import (
    MULTI_CONSTRAINT_COST_VERSION,
    HalfCheetahMultiConstraintCost,
)
from envs.half_cheetah_pitch_cost import (
    COST_VERSION as PITCH_COST_VERSION,
    HalfCheetahPitchCost,
)


ENVIRONMENT_CONFIGS = {
    "HalfCheetahPitchCost": {
        "factory": HalfCheetahPitchCost,
        "cost_version": PITCH_COST_VERSION,
        "constraint_names": ("pitch",),
    },
    "HalfCheetahJointSpeedCost": {
        "factory": HalfCheetahJointSpeedCost,
        "cost_version": JOINT_SPEED_COST_VERSION,
        "constraint_names": ("joint_speed",),
    },
    "HalfCheetahCombinedMaxCost": {
        "factory": HalfCheetahCombinedMaxCost,
        "cost_version": COMBINED_COST_VERSION,
        "constraint_names": ("combined_max",),
    },
    "HalfCheetahMultiConstraintCost": {
        "factory": HalfCheetahMultiConstraintCost,
        "cost_version": MULTI_CONSTRAINT_COST_VERSION,
        "constraint_names": ("pitch", "joint_speed"),
    },
}


class ActorGaussian(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action, activation):
        super().__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.activate_func = nn.ReLU() if activation == "relu" else nn.Tanh()

    def forward(self, state):
        state = self.activate_func(self.fc1(state))
        state = self.activate_func(self.fc2(state))
        state = self.activate_func(self.fc3(state))
        return self.max_action * torch.tanh(self.mean_layer(state))

    def deterministic_action(self, state):
        return self.forward(state)


class ActorBeta(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action, activation):
        super().__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.alpha_layer = nn.Linear(hidden_width, action_dim)
        self.beta_layer = nn.Linear(hidden_width, action_dim)
        self.activate_func = nn.ReLU() if activation == "relu" else nn.Tanh()

    def forward(self, state):
        state = self.activate_func(self.fc1(state))
        state = self.activate_func(self.fc2(state))
        alpha = F.softplus(self.alpha_layer(state)) + 1.0
        beta = F.softplus(self.beta_layer(state)) + 1.0
        return alpha, beta

    def deterministic_action(self, state):
        alpha, beta = self.forward(state)
        mean = alpha / (alpha + beta)
        return 2.0 * (mean - 0.5) * self.max_action


class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, value):
        value = np.asarray(value)
        self.n += 1
        if self.n == 1:
            self.mean = value
            self.std = value
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (value - old_mean) / self.n
            self.S = self.S + (value - old_mean) * (value - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape)

    def __call__(self, value, update=False):
        if update:
            self.running_ms.update(value)
        return (value - self.running_ms.mean) / (self.running_ms.std + 1e-8)


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "RunningMeanStd":
            return RunningMeanStd
        if name == "Normalization":
            return Normalization
        return super().find_class(module, name)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=2301)
    parser.add_argument(
        "--env",
        type=str,
        default="all",
        choices=("all", *ENVIRONMENT_CONFIGS.keys()),
    )
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--min_training_episodes", type=int, default=10000)
    parser.add_argument(
        "--checkpoint_mode",
        choices=("latest", "all"),
        default="latest",
    )
    parser.add_argument("--include_step_zero", action="store_true")
    parser.add_argument("--persistent_eps", type=float, default=None)
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_episode_steps", type=int, default=0)
    parser.add_argument(
        "--output_root",
        type=str,
        default="analysis/outputs/best_rcac_extended_evaluation",
    )
    return parser.parse_args()


def reset_observation(environment, seed):
    result = environment.reset(seed=seed)
    while isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float32)


def checkpoint_details(actor_path):
    prefix = "Best_RCAC_actor"
    suffix = actor_path.name[len(prefix):]
    if not suffix:
        return "best", None
    value = suffix[1:] if suffix.startswith("_") else suffix
    try:
        step = int(value)
        return f"step_{step}", step
    except ValueError:
        return value, None


def actor_sort_key(actor_path):
    checkpoint_name, checkpoint_step = checkpoint_details(actor_path)
    return checkpoint_step is None, checkpoint_step or -1, checkpoint_name


def discover_policies(repository_root, args):
    data_root = repository_root / "data_train"
    models_root = repository_root / "models"
    policies = []
    skipped = []
    for metadata_path in sorted(data_root.rglob("training_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as error:
            skipped.append({"path": str(metadata_path), "reason": str(error)})
            continue
        environment_name = metadata.get("environment")
        if environment_name not in ENVIRONMENT_CONFIGS:
            continue
        if args.env != "all" and environment_name != args.env:
            continue
        expected_version = ENVIRONMENT_CONFIGS[environment_name]["cost_version"]
        if metadata.get("cost_version") != expected_version:
            skipped.append(
                {
                    "path": str(metadata_path),
                    "reason": "cost_version does not match the current environment",
                }
            )
            continue
        try:
            run_number = int(metadata.get("run"))
            training_episodes = int(metadata.get("training_episodes", 0))
        except (TypeError, ValueError):
            skipped.append(
                {"path": str(metadata_path), "reason": "invalid run metadata"}
            )
            continue
        if args.run is not None and run_number != args.run:
            continue
        if training_episodes < args.min_training_episodes:
            skipped.append(
                {
                    "path": str(metadata_path),
                    "reason": f"training_episodes is below {args.min_training_episodes}",
                }
            )
            continue
        relative_run_directory = metadata_path.parent.relative_to(data_root)
        model_directory = models_root / relative_run_directory
        actor_paths = [
            path
            for path in model_directory.glob("Best_RCAC_actor*")
            if path.is_file()
            and (
                path.name == "Best_RCAC_actor"
                or path.name.startswith("Best_RCAC_actor_")
            )
        ]
        if not actor_paths:
            skipped.append(
                {
                    "path": str(model_directory),
                    "reason": "no saved Best_RCAC actor was found",
                }
            )
            continue
        for actor_path in sorted(actor_paths, key=actor_sort_key):
            checkpoint_name, checkpoint_step = checkpoint_details(actor_path)
            policy_id = (
                f"{environment_name}__{expected_version}__run{run_number}__"
                f"{checkpoint_name}"
            )
            policies.append(
                {
                    "policy_id": policy_id,
                    "environment": environment_name,
                    "cost_version": expected_version,
                    "run": run_number,
                    "training_seed": int(metadata.get("training_seed", -1)),
                    "training_episodes": training_episodes,
                    "checkpoint": checkpoint_name,
                    "checkpoint_step": checkpoint_step,
                    "actor_path": actor_path,
                    "metadata_path": metadata_path,
                    "metadata": metadata,
                }
            )
    return policies, skipped


def select_policies(policies, args):
    selected = [
        policy
        for policy in policies
        if args.include_step_zero or policy["checkpoint_step"] != 0
    ]
    if args.checkpoint_mode == "all":
        return selected
    grouped = {}
    for policy in selected:
        key = (
            policy["environment"],
            policy["cost_version"],
            policy["run"],
            policy["training_seed"],
        )
        grouped.setdefault(key, []).append(policy)
    latest = []
    for group in grouped.values():
        numbered = [
            policy for policy in group if policy["checkpoint_step"] is not None
        ]
        if numbered:
            latest.append(max(numbered, key=lambda policy: policy["checkpoint_step"]))
        else:
            latest.append(sorted(group, key=lambda policy: policy["policy_id"])[-1])
    return sorted(
        latest,
        key=lambda policy: (
            policy["environment"],
            policy["run"],
            policy["checkpoint_step"] if policy["checkpoint_step"] is not None else -1,
        ),
    )


def load_torch_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_normalizer(actor_path):
    normalizer_name = actor_path.name.replace("_actor", "_state_norm", 1)
    normalizer_path = actor_path.with_name(normalizer_name)
    if not normalizer_path.exists():
        return None, None
    with normalizer_path.open("rb") as handle:
        return CompatibleUnpickler(handle).load(), normalizer_path


def create_actor(policy, environment, args, device):
    metadata = policy["metadata"]
    state_dim = int(metadata.get("state_dimension", environment.observation_space.shape[0]))
    action_dim = int(metadata.get("action_dimension", environment.action_space.shape[0]))
    hidden_width = int(metadata.get("hidden_width", 64))
    actual_state_dim = int(environment.observation_space.shape[0])
    actual_action_dim = int(environment.action_space.shape[0])
    if state_dim != actual_state_dim or action_dim != actual_action_dim:
        raise ValueError(
            f"{policy['policy_id']} metadata dimensions ({state_dim}, {action_dim}) "
            f"do not match the environment ({actual_state_dim}, {actual_action_dim})"
        )
    max_action = float(np.asarray(environment.action_space.high).reshape(-1)[0])
    distribution = str(metadata.get("policy_distribution", "Gaussian"))
    if distribution == "Gaussian":
        actor = ActorGaussian(
            state_dim,
            action_dim,
            hidden_width,
            max_action,
            args.activation,
        )
    elif distribution == "Beta":
        actor = ActorBeta(
            state_dim,
            action_dim,
            hidden_width,
            max_action,
            args.activation,
        )
    else:
        raise ValueError(
            f"{policy['policy_id']} uses unsupported policy distribution {distribution}"
        )
    actor.load_state_dict(load_torch_state(policy["actor_path"], device))
    actor.to(device)
    actor.eval()
    return actor, max_action, distribution


def finite_values(values):
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def descriptive_statistics(values):
    array = finite_values(values)
    if array.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "median": math.nan,
            "minimum": math.nan,
            "maximum": math.nan,
        }
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    sem = std / math.sqrt(array.size)
    margin = 1.96 * sem
    return {
        "count": int(array.size),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def wilson_interval(successes, observations):
    if observations == 0:
        return math.nan, math.nan
    z = 1.96
    proportion = successes / observations
    denominator = 1.0 + z * z / observations
    center = (proportion + z * z / (2.0 * observations)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / observations
            + z * z / (4.0 * observations * observations)
        )
        / denominator
    )
    return center - margin, center + margin


def add_statistics(target, prefix, values):
    for key, value in descriptive_statistics(values).items():
        target[f"{prefix}_{key}"] = value


def component_value(info, name):
    if name == "pitch":
        value = info.get("pitch_cost", math.nan)
    elif name == "joint_speed":
        value = info.get("joint_speed_cost", math.nan)
    else:
        value = info.get("combined_cost", math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def evaluate_episode(
    environment,
    actor,
    state_normalizer,
    policy,
    episode_index,
    evaluation_seed,
    persistent_eps,
    max_action,
    max_episode_steps,
    device,
):
    state = reset_observation(environment, evaluation_seed)
    if state_normalizer is not None:
        state = np.asarray(state_normalizer(state, update=False), dtype=np.float32)
    reward_total = 0.0
    constraint_totals = None
    constraint_maximums = None
    pitch_total = 0.0
    joint_speed_total = 0.0
    combined_total = 0.0
    pitch_maximum = -math.inf
    joint_speed_maximum = -math.inf
    combined_maximum = -math.inf
    pitch_available = False
    joint_speed_available = False
    combined_available = False
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            action = actor.deterministic_action(state_tensor).cpu().numpy()[0]
        action = np.clip(action, -max_action, max_action)
        next_state, reward, cost, truncated, terminated, info = environment.step(action)
        truncated = bool(truncated)
        terminated = bool(terminated)
        cost_vector = np.maximum(
            np.asarray(cost, dtype=np.float64).reshape(-1),
            0.0,
        )
        if constraint_totals is None:
            constraint_totals = np.zeros(cost_vector.size, dtype=np.float64)
            constraint_maximums = np.full(cost_vector.size, -math.inf, dtype=np.float64)
        if cost_vector.size != constraint_totals.size:
            raise ValueError(f"Cost dimension changed during {policy['policy_id']}")
        constraint_totals += cost_vector
        constraint_maximums = np.maximum(constraint_maximums, cost_vector)
        pitch_cost = component_value(info, "pitch")
        joint_speed_cost = component_value(info, "joint_speed")
        combined_cost = component_value(info, "combined")
        if math.isfinite(pitch_cost):
            pitch_available = True
            pitch_total += pitch_cost
            pitch_maximum = max(pitch_maximum, pitch_cost)
        if math.isfinite(joint_speed_cost):
            joint_speed_available = True
            joint_speed_total += joint_speed_cost
            joint_speed_maximum = max(joint_speed_maximum, joint_speed_cost)
        if math.isfinite(combined_cost):
            combined_available = True
            combined_total += combined_cost
            combined_maximum = max(combined_maximum, combined_cost)
        reward_total += float(reward)
        steps += 1
        state = np.asarray(next_state, dtype=np.float32)
        if state_normalizer is not None:
            state = np.asarray(state_normalizer(state, update=False), dtype=np.float32)
        if max_episode_steps > 0 and steps >= max_episode_steps:
            truncated = True
    if constraint_totals is None or constraint_maximums is None:
        raise RuntimeError(f"{policy['policy_id']} produced an empty episode")
    constraint_names = ENVIRONMENT_CONFIGS[policy["environment"]]["constraint_names"]
    if len(constraint_names) != constraint_totals.size:
        raise ValueError(
            f"{policy['policy_id']} returned {constraint_totals.size} cost dimensions "
            f"but {len(constraint_names)} were expected"
        )
    if policy["environment"] in (
        "HalfCheetahCombinedMaxCost",
        "HalfCheetahMultiConstraintCost",
    ) and not (pitch_available and joint_speed_available and combined_available):
        raise ValueError(
            f"{policy['policy_id']} did not expose pitch_cost, joint_speed_cost, "
            "and combined_cost through info"
        )
    worst_cost = float(np.max(constraint_maximums))
    episode_feasible = bool(np.all(constraint_maximums <= persistent_eps))
    pitch_violation = bool(pitch_available and pitch_maximum > persistent_eps)
    joint_speed_violation = bool(
        joint_speed_available and joint_speed_maximum > persistent_eps
    )
    if pitch_violation and joint_speed_violation:
        violation_source = "pitch_and_joint_speed"
    elif pitch_violation:
        violation_source = "pitch"
    elif joint_speed_violation:
        violation_source = "joint_speed"
    else:
        violation_source = "none"
    row = {
        "policy_id": policy["policy_id"],
        "environment": policy["environment"],
        "cost_version": policy["cost_version"],
        "run": policy["run"],
        "training_seed": policy["training_seed"],
        "training_episodes": policy["training_episodes"],
        "checkpoint": policy["checkpoint"],
        "checkpoint_step": policy["checkpoint_step"],
        "actor_path": str(policy["actor_path"]),
        "episode": episode_index + 1,
        "evaluation_seed": evaluation_seed,
        "steps": steps,
        "reward": reward_total,
        "objective_total_cost": float(np.sum(constraint_totals)),
        "worst_instantaneous_cost": worst_cost,
        "combined_total_cost": combined_total if combined_available else math.nan,
        "combined_max_cost": combined_maximum if combined_available else math.nan,
        "persistent_epsilon": persistent_eps,
        "feasible": episode_feasible,
        "pitch_violation": pitch_violation,
        "joint_speed_violation": joint_speed_violation,
        "violation_source": violation_source,
        "pitch_total_cost": pitch_total if pitch_available else math.nan,
        "pitch_max_cost": pitch_maximum if pitch_available else math.nan,
        "joint_speed_total_cost": joint_speed_total
        if joint_speed_available
        else math.nan,
        "joint_speed_max_cost": joint_speed_maximum
        if joint_speed_available
        else math.nan,
        "constraint_0_name": constraint_names[0],
        "constraint_0_total_cost": float(constraint_totals[0]),
        "constraint_0_max_cost": float(constraint_maximums[0]),
        "constraint_1_name": constraint_names[1]
        if len(constraint_names) > 1
        else "",
        "constraint_1_total_cost": float(constraint_totals[1])
        if constraint_totals.size > 1
        else math.nan,
        "constraint_1_max_cost": float(constraint_maximums[1])
        if constraint_maximums.size > 1
        else math.nan,
    }
    return row


def summarize_policy(policy, episode_rows, state_normalizer_path, persistent_eps):
    feasible_rows = [row for row in episode_rows if row["feasible"]]
    feasible_count = len(feasible_rows)
    episode_count = len(episode_rows)
    feasibility_low, feasibility_high = wilson_interval(feasible_count, episode_count)
    summary = {
        "policy_id": policy["policy_id"],
        "environment": policy["environment"],
        "cost_version": policy["cost_version"],
        "run": policy["run"],
        "training_seed": policy["training_seed"],
        "training_episodes": policy["training_episodes"],
        "checkpoint": policy["checkpoint"],
        "checkpoint_step": policy["checkpoint_step"],
        "actor_path": str(policy["actor_path"]),
        "state_normalizer_path": str(state_normalizer_path)
        if state_normalizer_path is not None
        else "",
        "evaluation_episodes": episode_count,
        "persistent_epsilon": persistent_eps,
        "feasible_episode_count": feasible_count,
        "feasible_episode_fraction": feasible_count / episode_count,
        "feasible_fraction_ci95_low": feasibility_low,
        "feasible_fraction_ci95_high": feasibility_high,
        "all_episodes_feasible": feasible_count == episode_count,
    }
    add_statistics(summary, "reward", [row["reward"] for row in episode_rows])
    add_statistics(
        summary,
        "feasible_reward",
        [row["reward"] for row in feasible_rows],
    )
    add_statistics(
        summary,
        "worst_instantaneous_cost",
        [row["worst_instantaneous_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "objective_total_cost",
        [row["objective_total_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "combined_total_cost",
        [row["combined_total_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "combined_max_cost",
        [row["combined_max_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "pitch_total_cost",
        [row["pitch_total_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "pitch_max_cost",
        [row["pitch_max_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "joint_speed_total_cost",
        [row["joint_speed_total_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "joint_speed_max_cost",
        [row["joint_speed_max_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "constraint_0_total_cost",
        [row["constraint_0_total_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "constraint_0_max_cost",
        [row["constraint_0_max_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "constraint_1_total_cost",
        [row["constraint_1_total_cost"] for row in episode_rows],
    )
    add_statistics(
        summary,
        "constraint_1_max_cost",
        [row["constraint_1_max_cost"] for row in episode_rows],
    )
    summary["pitch_violation_count"] = sum(
        row["pitch_violation"] for row in episode_rows
    )
    summary["pitch_violation_fraction"] = (
        summary["pitch_violation_count"] / episode_count
    )
    summary["joint_speed_violation_count"] = sum(
        row["joint_speed_violation"] for row in episode_rows
    )
    summary["joint_speed_violation_fraction"] = (
        summary["joint_speed_violation_count"] / episode_count
    )
    return summary


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows were available for {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_arguments()
    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if args.min_training_episodes < 0:
        raise ValueError("min_training_episodes cannot be negative")
    if args.max_episode_steps < 0:
        raise ValueError("max_episode_steps cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    repository_root = Path(__file__).resolve().parent
    discovered_policies, skipped = discover_policies(repository_root, args)
    policies = select_policies(discovered_policies, args)
    if not policies:
        raise FileNotFoundError(
            "No matching saved Best_RCAC actors were found under the models directory"
        )
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = output_root / f"evaluation_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=False)
    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    all_episode_rows = []
    all_summary_rows = []
    failures = []
    for policy_index, policy in enumerate(policies, start=1):
        environment = None
        try:
            environment = ENVIRONMENT_CONFIGS[policy["environment"]]["factory"]()
            environment.action_space.seed(args.base_seed)
            actor, max_action, distribution = create_actor(
                policy,
                environment,
                args,
                device,
            )
            state_normalizer, state_normalizer_path = load_state_normalizer(
                policy["actor_path"]
            )
            persistent_eps = (
                float(args.persistent_eps)
                if args.persistent_eps is not None
                else float(policy["metadata"].get("persistent_epsilon", 0.1))
            )
            episode_rows = []
            for episode_index in range(args.episodes):
                evaluation_seed = args.base_seed + episode_index
                episode_rows.append(
                    evaluate_episode(
                        environment,
                        actor,
                        state_normalizer,
                        policy,
                        episode_index,
                        evaluation_seed,
                        persistent_eps,
                        max_action,
                        args.max_episode_steps,
                        device,
                    )
                )
            summary = summarize_policy(
                policy,
                episode_rows,
                state_normalizer_path,
                persistent_eps,
            )
            summary["policy_distribution"] = distribution
            summary["activation"] = args.activation
            all_episode_rows.extend(episode_rows)
            all_summary_rows.append(summary)
            print(
                f"[{policy_index}/{len(policies)}] {policy['policy_id']}: "
                f"reward={summary['reward_mean']:.3f} "
                f"95%CI=[{summary['reward_ci95_low']:.3f}, "
                f"{summary['reward_ci95_high']:.3f}] "
                f"safe={summary['feasible_episode_count']}/{args.episodes} "
                f"max_cost={summary['worst_instantaneous_cost_maximum']:.6f}"
            )
        except Exception as error:
            failures.append(
                {
                    "policy_id": policy["policy_id"],
                    "actor_path": str(policy["actor_path"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"[{policy_index}/{len(policies)}] {policy['policy_id']}: "
                f"FAILED: {type(error).__name__}: {error}"
            )
        finally:
            if environment is not None:
                environment.close()
    if not all_summary_rows:
        failure_path = output_directory / "evaluation_metadata.json"
        failure_path.write_text(
            json.dumps(json_ready({"failures": failures, "skipped": skipped}), indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"Every policy evaluation failed. Details: {failure_path}")
    episode_path = output_directory / "episode_metrics.csv"
    summary_path = output_directory / "policy_summary.csv"
    metadata_path = output_directory / "evaluation_metadata.json"
    write_csv(episode_path, all_episode_rows)
    write_csv(summary_path, all_summary_rows)
    evaluation_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repository_root": str(repository_root),
        "episodes_per_policy": args.episodes,
        "base_seed": args.base_seed,
        "evaluation_seeds": [
            args.base_seed + episode_index for episode_index in range(args.episodes)
        ],
        "environment_filter": args.env,
        "run_filter": args.run,
        "minimum_training_episodes": args.min_training_episodes,
        "persistent_epsilon_override": args.persistent_eps,
        "activation": args.activation,
        "device": str(device),
        "max_episode_steps_override": args.max_episode_steps,
        "deterministic_actions": True,
        "checkpoint_mode": args.checkpoint_mode,
        "include_step_zero": args.include_step_zero,
        "discovered_policy_count": len(discovered_policies),
        "selected_policy_count": len(policies),
        "successful_policy_count": len(all_summary_rows),
        "failed_policy_count": len(failures),
        "skipped": skipped,
        "failures": failures,
        "episode_metrics": str(episode_path),
        "policy_summary": str(summary_path),
    }
    metadata_path.write_text(
        json.dumps(json_ready(evaluation_metadata), indent=2),
        encoding="utf-8",
    )
    print(f"Episode metrics: {episode_path}")
    print(f"Policy summary: {summary_path}")
    print(f"Evaluation metadata: {metadata_path}")


if __name__ == "__main__":
    main()