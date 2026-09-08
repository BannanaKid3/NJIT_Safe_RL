import argparse
import csv
import itertools
import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as error:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = error
else:
    TORCH_IMPORT_ERROR = None

from configs.halfcheetah_combined_safety import (
    JOINT_NAMES,
    JOINT_SPEED_SCALES,
    JOINT_SPEED_THRESHOLDS,
    PERSISTENT_EPS,
    PITCH_SCALE,
    PITCH_THRESHOLD,
)
try:
    from envs.half_cheetah import ACTION_TORQUE_THRESHOLD, HalfCheetahWithPos
except ModuleNotFoundError as error:
    ACTION_TORQUE_THRESHOLD = 0.5
    HalfCheetahWithPos = None
    ENV_IMPORT_ERROR = error
else:
    ENV_IMPORT_ERROR = None


METHOD_ORDER = ("Shilpa action", "Scalar combined", "Multi-constraint")
METHOD_COLORS = {
    "Shilpa action": "#6B7280",
    "Scalar combined": "#2F6BFF",
    "Multi-constraint": "#F28E2B",
}
SEED_MARKERS = {5: "o", 11: "s", 17: "^", 23: "D", 29: "P", 37: "X"}
SCALAR_VERSION = "combined_max_wrapped90_jointp99_v1"
MULTI_VERSION = "multi_constraint_pitch_joint_v1"


@dataclass(frozen=True)
class PolicySpec:
    method: str
    training_objective: str
    training_seed: int
    run: int
    actor_path: Path
    metadata_path: Path | None


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


if nn is not None:
    class ActorGaussian(nn.Module):
        def __init__(self, state_dim, hidden_width, action_dim, max_action, activation):
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
else:
    class ActorGaussian:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is required to load and evaluate the saved actors"
            ) from TORCH_IMPORT_ERROR


def parse_int_list(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")
    try:
        return [int(item) for item in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_int_list, default=[5, 11, 17, 23])
    parser.add_argument("--action_runs", type=parse_int_list, default=[10, 11, 12, 13])
    parser.add_argument("--scalar_runs", type=parse_int_list, default=[4, 5, 6, 7])
    parser.add_argument("--multi_runs", type=parse_int_list, default=[4, 5, 6, 7])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=2301)
    parser.add_argument("--plot_episode", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--persistent_eps", type=float, default=float(PERSISTENT_EPS))
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--analysis_seed", type=int, default=91021)
    parser.add_argument("--save_all_timesteps", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--output_root",
        type=str,
        default="analysis/outputs/unified_rcac_comparison",
    )
    return parser.parse_args()


def validate_arguments(args):
    lengths = {
        len(args.seeds),
        len(args.action_runs),
        len(args.scalar_runs),
        len(args.multi_runs),
    }
    if len(lengths) != 1:
        raise ValueError(
            "seeds, action_runs, scalar_runs, and multi_runs must have equal lengths"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be unique")
    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if not 1 <= args.plot_episode <= args.episodes:
        raise ValueError("plot_episode must be between 1 and episodes")
    if args.max_episode_steps < 1:
        raise ValueError("max_episode_steps must be at least 1")
    if args.persistent_eps < 0.0:
        raise ValueError("persistent_eps cannot be negative")
    if args.bootstrap_samples < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")
    if args.device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is unavailable")


def numbered_checkpoint(directory):
    candidates = []
    pattern = re.compile(r"^Best_RCAC_actor_(\d+)$")
    for path in directory.glob("Best_RCAC_actor_*"):
        match = pattern.match(path.name)
        if path.is_file() and match and int(match.group(1)) > 0:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    fallback = directory / "Best_RCAC_actor"
    return fallback


def build_policy_specs(repo_root, args):
    specs = []
    for seed, action_run, scalar_run, multi_run in zip(
        args.seeds,
        args.action_runs,
        args.scalar_runs,
        args.multi_runs,
    ):
        action_directory = repo_root / "models" / "HalfCheetahWithPos" / f"run{action_run}"
        scalar_directory = (
            repo_root
            / "models"
            / "HalfCheetahCombinedMaxCost"
            / SCALAR_VERSION
            / f"run{scalar_run}"
        )
        multi_directory = (
            repo_root
            / "models"
            / "HalfCheetahMultiConstraintCost"
            / MULTI_VERSION
            / f"run{multi_run}"
        )
        specs.extend(
            [
                PolicySpec(
                    "Shilpa action",
                    "action_cost",
                    seed,
                    action_run,
                    action_directory / "Best_RCAC_actor",
                    repo_root
                    / "data_train"
                    / "HalfCheetahWithPos"
                    / f"run{action_run}"
                    / "training_metadata.json",
                ),
                PolicySpec(
                    "Scalar combined",
                    "combined_state_cost",
                    seed,
                    scalar_run,
                    scalar_directory / "Best_RCAC_actor",
                    repo_root
                    / "data_train"
                    / "HalfCheetahCombinedMaxCost"
                    / SCALAR_VERSION
                    / f"run{scalar_run}"
                    / "training_metadata.json",
                ),
                PolicySpec(
                    "Multi-constraint",
                    "separate_pitch_and_joint_costs",
                    seed,
                    multi_run,
                    numbered_checkpoint(multi_directory),
                    repo_root
                    / "data_train"
                    / "HalfCheetahMultiConstraintCost"
                    / MULTI_VERSION
                    / f"run{multi_run}"
                    / "training_metadata.json",
                ),
            ]
        )
    return specs


def validate_policy_specs(specs):
    missing = [spec.actor_path for spec in specs if not spec.actor_path.exists()]
    if missing:
        lines = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing actor checkpoints:\n{lines}")
    errors = []
    for spec in specs:
        if spec.metadata_path is None or not spec.metadata_path.exists():
            continue
        try:
            metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"{spec.metadata_path}: {error}")
            continue
        metadata_seed = metadata.get("training_seed")
        metadata_run = metadata.get("run")
        if metadata_seed is not None and int(metadata_seed) != spec.training_seed:
            errors.append(
                f"{spec.metadata_path}: expected seed {spec.training_seed}, found {metadata_seed}"
            )
        if metadata_run is not None and int(metadata_run) != spec.run:
            errors.append(f"{spec.metadata_path}: expected run {spec.run}, found {metadata_run}")
    if errors:
        raise ValueError("Policy metadata validation failed:\n" + "\n".join(errors))


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "actor_state_dict" in state:
        state = state["actor_state_dict"]
    cleaned = {}
    for key, value in state.items():
        name = key
        for prefix in ("module.", "actor."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def load_actor(spec, max_action, activation, device):
    state = clean_state_dict(torch_load(spec.actor_path, device))
    required = {"fc1.weight", "fc2.weight", "fc3.weight", "mean_layer.weight"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"{spec.actor_path} is not the expected Gaussian actor: {missing}")
    state_dim = int(state["fc1.weight"].shape[1])
    hidden_width = int(state["fc1.weight"].shape[0])
    action_dim = int(state["mean_layer.weight"].shape[0])
    actor = ActorGaussian(state_dim, hidden_width, action_dim, max_action, activation)
    actor.load_state_dict(state, strict=True)
    actor.to(device)
    actor.eval()
    return actor, state_dim, action_dim, hidden_width


def normalizer_path_for_actor(actor_path):
    return actor_path.with_name(actor_path.name.replace("_actor", "_state_norm", 1))


def load_state_normalizer(actor_path):
    path = normalizer_path_for_actor(actor_path)
    if not path.exists():
        return None, None
    with path.open("rb") as handle:
        return CompatibleUnpickler(handle).load(), path


def reset_observation(environment, seed):
    np.random.seed(seed)
    result = environment.reset(seed=seed)
    while isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float32)


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def physical_values(environment, action):
    pitch_raw = float(environment.sim.data.qpos[2])
    pitch = wrap_angle(pitch_raw)
    pitch_abs = abs(pitch)
    joint_velocities = np.asarray(environment.sim.data.qvel[3:9], dtype=np.float64).copy()
    joint_speeds = np.abs(joint_velocities)
    action = np.asarray(action, dtype=np.float64)
    maximum_absolute_action = float(np.max(np.abs(action)))
    action_cost = max(maximum_absolute_action - float(ACTION_TORQUE_THRESHOLD), 0.0)
    pitch_cost = max(pitch_abs - float(PITCH_THRESHOLD), 0.0) / float(PITCH_SCALE)
    thresholds = np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64)
    scales = np.asarray(JOINT_SPEED_SCALES, dtype=np.float64)
    joint_costs = np.maximum(joint_speeds - thresholds, 0.0) / scales
    joint_speed_cost = float(np.max(joint_costs))
    combined_state_cost = max(float(pitch_cost), joint_speed_cost)
    return {
        "pitch_raw": pitch_raw,
        "pitch": pitch,
        "pitch_abs": pitch_abs,
        "pitch_cost": float(pitch_cost),
        "joint_velocities": joint_velocities,
        "joint_speeds": joint_speeds,
        "joint_costs": joint_costs,
        "joint_speed_cost": joint_speed_cost,
        "combined_state_cost": combined_state_cost,
        "maximum_absolute_action": maximum_absolute_action,
        "action_cost": float(action_cost),
    }


def percentile(values, quantile):
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def episode_summary(spec, episode_index, evaluation_seed, arrays, persistent_eps):
    rewards = arrays["rewards"]
    action_costs = arrays["action_costs"]
    maximum_actions = arrays["maximum_actions"]
    pitch_values = arrays["pitch_abs"]
    pitch_costs = arrays["pitch_costs"]
    joint_speeds = arrays["joint_speeds"]
    joint_costs = arrays["joint_costs"]
    joint_max_costs = arrays["joint_speed_costs"]
    combined_costs = arrays["combined_state_costs"]
    row = {
        "method": spec.method,
        "training_objective": spec.training_objective,
        "training_seed": spec.training_seed,
        "run": spec.run,
        "actor_path": str(spec.actor_path),
        "episode": episode_index,
        "evaluation_seed": evaluation_seed,
        "steps": len(rewards),
        "reward": float(np.sum(rewards)),
        "maximum_absolute_action": float(np.max(maximum_actions)),
        "mean_maximum_absolute_action": float(np.mean(maximum_actions)),
        "action_cost_total": float(np.sum(action_costs)),
        "action_cost_max": float(np.max(action_costs)),
        "action_cost_positive_timestep_fraction": float(np.mean(action_costs > 0.0)),
        "action_violation_timestep_fraction": float(np.mean(action_costs > persistent_eps)),
        "action_feasible": bool(np.max(action_costs) <= persistent_eps),
        "pitch_abs_mean_rad": float(np.mean(pitch_values)),
        "pitch_abs_rms_rad": float(np.sqrt(np.mean(np.square(pitch_values)))),
        "pitch_abs_p95_rad": percentile(pitch_values, 0.95),
        "pitch_abs_p99_rad": percentile(pitch_values, 0.99),
        "pitch_abs_max_rad": float(np.max(pitch_values)),
        "pitch_abs_max_deg": float(np.degrees(np.max(pitch_values))),
        "pitch_max_utilization": float(np.max(pitch_values) / float(PITCH_THRESHOLD)),
        "pitch_cost_total": float(np.sum(pitch_costs)),
        "pitch_cost_max": float(np.max(pitch_costs)),
        "pitch_violation_timestep_fraction": float(np.mean(pitch_costs > persistent_eps)),
        "pitch_feasible": bool(np.max(pitch_costs) <= persistent_eps),
        "joint_speed_cost_total": float(np.sum(joint_max_costs)),
        "joint_speed_cost_max": float(np.max(joint_max_costs)),
        "joint_violation_timestep_fraction": float(np.mean(joint_max_costs > persistent_eps)),
        "joint_feasible": bool(np.max(joint_max_costs) <= persistent_eps),
        "combined_state_cost_total": float(np.sum(combined_costs)),
        "combined_state_cost_max": float(np.max(combined_costs)),
        "state_feasible": bool(np.max(combined_costs) <= persistent_eps),
    }
    thresholds = np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64)
    for index, name in enumerate(JOINT_NAMES):
        speeds = joint_speeds[:, index]
        costs = joint_costs[:, index]
        row[f"{name}_speed_mean"] = float(np.mean(speeds))
        row[f"{name}_speed_rms"] = float(np.sqrt(np.mean(np.square(speeds))))
        row[f"{name}_speed_p95"] = percentile(speeds, 0.95)
        row[f"{name}_speed_p99"] = percentile(speeds, 0.99)
        row[f"{name}_speed_max"] = float(np.max(speeds))
        row[f"{name}_max_utilization"] = float(np.max(speeds) / thresholds[index])
        row[f"{name}_cost_total"] = float(np.sum(costs))
        row[f"{name}_cost_max"] = float(np.max(costs))
    row["joint_max_utilization"] = max(
        row[f"{name}_max_utilization"] for name in JOINT_NAMES
    )
    row["all_constraints_feasible"] = bool(
        row["action_feasible"] and row["pitch_feasible"] and row["joint_feasible"]
    )
    if spec.training_objective == "action_cost":
        row["own_objective_feasible"] = row["action_feasible"]
    else:
        row["own_objective_feasible"] = row["state_feasible"]
    return row


def timestep_row(spec, episode_index, evaluation_seed, timestep, time_seconds, reward, action, values):
    row = {
        "method": spec.method,
        "training_objective": spec.training_objective,
        "training_seed": spec.training_seed,
        "run": spec.run,
        "episode": episode_index,
        "evaluation_seed": evaluation_seed,
        "timestep": timestep,
        "time_seconds": time_seconds,
        "reward": float(reward),
        "maximum_absolute_action": values["maximum_absolute_action"],
        "action_cost": values["action_cost"],
        "torso_pitch_raw": values["pitch_raw"],
        "torso_pitch": values["pitch"],
        "torso_pitch_abs": values["pitch_abs"],
        "pitch_cost": values["pitch_cost"],
        "joint_speed_cost": values["joint_speed_cost"],
        "combined_state_cost": values["combined_state_cost"],
    }
    for index, name in enumerate(JOINT_NAMES):
        row[f"action_{index + 1}"] = float(action[index])
        row[f"{name}_velocity"] = float(values["joint_velocities"][index])
        row[f"{name}_speed"] = float(values["joint_speeds"][index])
        row[f"{name}_cost"] = float(values["joint_costs"][index])
    return row


def evaluate_policy(
    spec,
    args,
    device,
    selected_timestep_rows,
    all_timestep_writer,
):
    environment = HalfCheetahWithPos()
    environment.action_space.seed(args.base_seed)
    max_action = float(np.asarray(environment.action_space.high).reshape(-1)[0])
    actor, state_dim, action_dim, hidden_width = load_actor(
        spec,
        max_action,
        args.activation,
        device,
    )
    expected_state_dim = int(environment.observation_space.shape[0])
    expected_action_dim = int(environment.action_space.shape[0])
    if state_dim != expected_state_dim or action_dim != expected_action_dim:
        environment.close()
        raise ValueError(
            f"{spec.actor_path} dimensions ({state_dim}, {action_dim}) do not match "
            f"HalfCheetahWithPos ({expected_state_dim}, {expected_action_dim})"
        )
    state_normalizer, state_normalizer_path = load_state_normalizer(spec.actor_path)
    episode_rows = []
    for episode_index in range(1, args.episodes + 1):
        evaluation_seed = args.base_seed + episode_index - 1
        observation = reset_observation(environment, evaluation_seed)
        arrays = {
            "rewards": [],
            "action_costs": [],
            "maximum_actions": [],
            "pitch_abs": [],
            "pitch_costs": [],
            "joint_speeds": [],
            "joint_costs": [],
            "joint_speed_costs": [],
            "combined_state_costs": [],
        }
        done = False
        timestep = 0
        while not done and timestep < args.max_episode_steps:
            actor_observation = observation
            if state_normalizer is not None:
                actor_observation = state_normalizer(actor_observation, update=False)
            state_tensor = torch.as_tensor(
                actor_observation,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().reshape(-1)
            action = np.clip(action, environment.action_space.low, environment.action_space.high)
            next_observation, reward, environment_action_cost, truncated, terminated, _ = environment.step(action)
            done = bool(truncated or terminated)
            values = physical_values(environment, action)
            returned_cost = float(np.asarray(environment_action_cost).reshape(-1)[0])
            if not np.isclose(returned_cost, values["action_cost"], rtol=0.0, atol=1e-9):
                environment.close()
                raise ValueError(
                    f"Action-cost mismatch for {spec.method}, seed {spec.training_seed}: "
                    f"environment={returned_cost}, recomputed={values['action_cost']}"
                )
            arrays["rewards"].append(float(reward))
            arrays["action_costs"].append(values["action_cost"])
            arrays["maximum_actions"].append(values["maximum_absolute_action"])
            arrays["pitch_abs"].append(values["pitch_abs"])
            arrays["pitch_costs"].append(values["pitch_cost"])
            arrays["joint_speeds"].append(values["joint_speeds"])
            arrays["joint_costs"].append(values["joint_costs"])
            arrays["joint_speed_costs"].append(values["joint_speed_cost"])
            arrays["combined_state_costs"].append(values["combined_state_cost"])
            timestep += 1
            row = timestep_row(
                spec,
                episode_index,
                evaluation_seed,
                timestep,
                float(timestep * environment.dt),
                reward,
                action,
                values,
            )
            if episode_index == args.plot_episode:
                selected_timestep_rows.append(row)
            if all_timestep_writer is not None:
                all_timestep_writer.writerow(row)
            observation = np.asarray(next_observation, dtype=np.float32)
        for key in arrays:
            arrays[key] = np.asarray(arrays[key], dtype=np.float64)
        summary = episode_summary(
            spec,
            episode_index,
            evaluation_seed,
            arrays,
            args.persistent_eps,
        )
        summary["state_normalizer_path"] = (
            str(state_normalizer_path) if state_normalizer_path is not None else ""
        )
        summary["hidden_width"] = hidden_width
        episode_rows.append(summary)
        print(
            f"{spec.method} seed={spec.training_seed} "
            f"[{episode_index}/{args.episodes}] reward={summary['reward']:.3f} "
            f"action_safe={summary['action_feasible']} "
            f"state_safe={summary['state_feasible']}"
        )
    environment.close()
    return episode_rows


def z_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean, 0.0
    standard_deviation = float(np.std(values, ddof=1))
    margin = 1.96 * standard_deviation / math.sqrt(values.size)
    return mean, mean - margin, mean + margin, standard_deviation


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


T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean, 0.0
    standard_deviation = float(np.std(values, ddof=1))
    critical = T_CRITICAL_975.get(values.size - 1, 1.96)
    margin = critical * standard_deviation / math.sqrt(values.size)
    return mean, mean - margin, mean + margin, standard_deviation


def summarize_policies(episode_frame):
    rows = []
    boolean_metrics = (
        "action_feasible",
        "pitch_feasible",
        "joint_feasible",
        "state_feasible",
        "all_constraints_feasible",
        "own_objective_feasible",
    )
    numeric_metrics = (
        "reward",
        "maximum_absolute_action",
        "mean_maximum_absolute_action",
        "action_cost_total",
        "action_cost_max",
        "action_cost_positive_timestep_fraction",
        "action_violation_timestep_fraction",
        "pitch_abs_mean_rad",
        "pitch_abs_rms_rad",
        "pitch_abs_p95_rad",
        "pitch_abs_p99_rad",
        "pitch_abs_max_rad",
        "pitch_abs_max_deg",
        "pitch_max_utilization",
        "pitch_cost_total",
        "pitch_cost_max",
        "joint_speed_cost_total",
        "joint_speed_cost_max",
        "joint_max_utilization",
        "combined_state_cost_total",
        "combined_state_cost_max",
    )
    numeric_metrics += tuple(
        f"{name}_{suffix}"
        for name in JOINT_NAMES
        for suffix in (
            "speed_mean",
            "speed_rms",
            "speed_p95",
            "speed_p99",
            "speed_max",
            "max_utilization",
            "cost_total",
            "cost_max",
        )
    )
    for (method, training_seed, run), group in episode_frame.groupby(
        ["method", "training_seed", "run"], sort=False
    ):
        row = {
            "method": method,
            "training_objective": group["training_objective"].iloc[0],
            "training_seed": int(training_seed),
            "run": int(run),
            "actor_path": group["actor_path"].iloc[0],
            "evaluation_episodes": len(group),
        }
        for metric in numeric_metrics:
            mean, low, high, standard_deviation = z_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = standard_deviation
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_minimum"] = float(group[metric].min())
            row[f"{metric}_maximum"] = float(group[metric].max())
        for metric in boolean_metrics:
            count = int(group[metric].astype(bool).sum())
            low, high = wilson_interval(count, len(group))
            row[f"{metric}_count"] = count
            row[f"{metric}_fraction"] = count / len(group)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    result = pd.DataFrame(rows)
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result["_method_rank"] = result["method"].map(method_rank)
    result = result.sort_values(["training_seed", "_method_rank"]).drop(columns="_method_rank")
    return result.reset_index(drop=True)


def summarize_methods(policy_frame):
    metrics = (
        "reward_mean",
        "action_feasible_fraction",
        "pitch_feasible_fraction",
        "joint_feasible_fraction",
        "state_feasible_fraction",
        "all_constraints_feasible_fraction",
        "own_objective_feasible_fraction",
        "maximum_absolute_action_mean",
        "pitch_abs_max_deg_mean",
        "pitch_max_utilization_mean",
        "joint_max_utilization_mean",
    )
    rows = []
    for method in METHOD_ORDER:
        group = policy_frame[policy_frame["method"] == method]
        if group.empty:
            continue
        row = {"method": method, "training_seed_count": len(group)}
        for metric in metrics:
            mean, low, high, standard_deviation = t_ci(group[metric])
            row[f"{metric}_across_seed_mean"] = mean
            row[f"{metric}_across_seed_sd"] = standard_deviation
            row[f"{metric}_across_seed_ci95_low"] = low
            row[f"{metric}_across_seed_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(differences):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.abs(differences) > 1e-15]
    if differences.size == 0:
        return 1.0
    observed = abs(float(np.mean(differences)))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def bootstrap_paired_ci(differences, samples, rng):
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size == 1:
        value = float(differences[0])
        return value, value
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def seed_level_pairwise_tests(policy_frame, args):
    metrics = {
        "reward_mean": "higher",
        "action_feasible_fraction": "higher",
        "state_feasible_fraction": "higher",
        "all_constraints_feasible_fraction": "higher",
        "pitch_max_utilization_mean": "lower",
        "joint_max_utilization_mean": "lower",
        "maximum_absolute_action_mean": "lower",
    }
    rng = np.random.default_rng(args.analysis_seed)
    rows = []
    for method_a, method_b in itertools.combinations(METHOD_ORDER, 2):
        first = policy_frame[policy_frame["method"] == method_a]
        second = policy_frame[policy_frame["method"] == method_b]
        merged = first.merge(second, on="training_seed", suffixes=("_a", "_b"))
        for metric, preferred_direction in metrics.items():
            differences = merged[f"{metric}_a"].to_numpy() - merged[f"{metric}_b"].to_numpy()
            mean, low, high, standard_deviation = t_ci(differences)
            bootstrap_low, bootstrap_high = bootstrap_paired_ci(
                differences,
                args.bootstrap_samples,
                rng,
            )
            if preferred_direction == "higher":
                a_wins = int(np.sum(differences > 0.0))
                b_wins = int(np.sum(differences < 0.0))
            else:
                a_wins = int(np.sum(differences < 0.0))
                b_wins = int(np.sum(differences > 0.0))
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "preferred_direction": preferred_direction,
                    "matched_training_seeds": len(merged),
                    "seed_list": ",".join(str(int(seed)) for seed in merged["training_seed"]),
                    "mean_difference_a_minus_b": mean,
                    "difference_sd": standard_deviation,
                    "t_ci95_low": low,
                    "t_ci95_high": high,
                    "bootstrap_ci95_low": bootstrap_low,
                    "bootstrap_ci95_high": bootstrap_high,
                    "exact_sign_flip_pvalue": exact_sign_flip_pvalue(differences),
                    "method_a_seed_wins": a_wins,
                    "method_b_seed_wins": b_wins,
                    "ties": int(np.sum(np.abs(differences) <= 1e-15)),
                }
            )
    return pd.DataFrame(rows)


def exact_mcnemar_pvalue(a_only, b_only):
    discordant = int(a_only + b_only)
    if discordant == 0:
        return 1.0
    smaller = min(int(a_only), int(b_only))
    probability = sum(math.comb(discordant, index) for index in range(smaller + 1))
    probability /= 2**discordant
    return min(1.0, 2.0 * probability)


def within_seed_pairwise_tests(episode_frame):
    continuous_metrics = (
        "reward",
        "pitch_abs_max_deg",
        "pitch_max_utilization",
        "joint_max_utilization",
        "maximum_absolute_action",
    )
    feasibility_metrics = (
        "action_feasible",
        "state_feasible",
        "all_constraints_feasible",
    )
    rows = []
    for training_seed in sorted(episode_frame["training_seed"].unique()):
        seed_frame = episode_frame[episode_frame["training_seed"] == training_seed]
        for method_a, method_b in itertools.combinations(METHOD_ORDER, 2):
            first = seed_frame[seed_frame["method"] == method_a]
            second = seed_frame[seed_frame["method"] == method_b]
            merged = first.merge(second, on="evaluation_seed", suffixes=("_a", "_b"))
            for metric in continuous_metrics:
                differences = merged[f"{metric}_a"].to_numpy() - merged[f"{metric}_b"].to_numpy()
                mean, low, high, standard_deviation = t_ci(differences)
                rows.append(
                    {
                        "training_seed": int(training_seed),
                        "method_a": method_a,
                        "method_b": method_b,
                        "test_type": "paired_reset_difference",
                        "metric": metric,
                        "matched_resets": len(merged),
                        "mean_difference_a_minus_b": mean,
                        "difference_sd": standard_deviation,
                        "ci95_low": low,
                        "ci95_high": high,
                        "a_safe_b_unsafe": math.nan,
                        "a_unsafe_b_safe": math.nan,
                        "exact_mcnemar_pvalue": math.nan,
                    }
                )
            for metric in feasibility_metrics:
                a = merged[f"{metric}_a"].astype(bool)
                b = merged[f"{metric}_b"].astype(bool)
                a_only = int((a & ~b).sum())
                b_only = int((~a & b).sum())
                rows.append(
                    {
                        "training_seed": int(training_seed),
                        "method_a": method_a,
                        "method_b": method_b,
                        "test_type": "matched_reset_mcnemar",
                        "metric": metric,
                        "matched_resets": len(merged),
                        "mean_difference_a_minus_b": float(a.mean() - b.mean()),
                        "difference_sd": math.nan,
                        "ci95_low": math.nan,
                        "ci95_high": math.nan,
                        "a_safe_b_unsafe": a_only,
                        "a_unsafe_b_safe": b_only,
                        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(a_only, b_only),
                    }
                )
    return pd.DataFrame(rows)


def pareto_rows(policy_frame):
    rows = []
    for training_seed in sorted(policy_frame["training_seed"].unique()):
        seed_frame = policy_frame[policy_frame["training_seed"] == training_seed]
        for safety_metric, comparison_name in (
            ("state_feasible_fraction", "reward_vs_state_feasibility"),
            ("all_constraints_feasible_fraction", "reward_vs_all_constraint_feasibility"),
        ):
            for _, current in seed_frame.iterrows():
                dominators = []
                for _, other in seed_frame.iterrows():
                    if other["method"] == current["method"]:
                        continue
                    no_worse = (
                        other["reward_mean"] >= current["reward_mean"]
                        and other[safety_metric] >= current[safety_metric]
                    )
                    strictly_better = (
                        other["reward_mean"] > current["reward_mean"]
                        or other[safety_metric] > current[safety_metric]
                    )
                    if no_worse and strictly_better:
                        dominators.append(other["method"])
                rows.append(
                    {
                        "training_seed": int(training_seed),
                        "comparison": comparison_name,
                        "method": current["method"],
                        "reward_mean": current["reward_mean"],
                        "safety_metric": safety_metric,
                        "safety_value": current[safety_metric],
                        "pareto_nondominated": not dominators,
                        "dominated_by": ", ".join(dominators),
                    }
                )
    return pd.DataFrame(rows)


def method_positions(methods):
    return np.arange(len(methods), dtype=np.float64)


def add_seed_legend(axis, seeds):
    handles = []
    for seed in seeds:
        handle = plt.Line2D(
            [],
            [],
            color="black",
            marker=SEED_MARKERS.get(int(seed), "o"),
            linestyle="None",
            markersize=7,
            label=f"seed {int(seed)}",
        )
        handles.append(handle)
    return handles


def save_figure(figure, path, manifest, description):
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    manifest.append({"figure": path.name, "description": description})


def plot_reward_by_seed(policy_frame, figure_directory, manifest):
    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    for method in METHOD_ORDER:
        group = policy_frame[policy_frame["method"] == method].sort_values("training_seed")
        if group.empty:
            continue
        yerr = np.vstack(
            [
                group["reward_mean"] - group["reward_ci95_low"],
                group["reward_ci95_high"] - group["reward_mean"],
            ]
        )
        axis.errorbar(
            group["training_seed"],
            group["reward_mean"],
            yerr=yerr,
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=method,
        )
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Mean reward over evaluation resets")
    axis.set_title("Final Policy Reward by Training Seed")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(
        figure,
        figure_directory / "reward_by_training_seed.png",
        manifest,
        "Mean 100-episode reward and reset-level 95% confidence interval for each policy.",
    )


def plot_feasibility_by_seed(policy_frame, figure_directory, manifest):
    panels = (
        ("own_objective_feasible_fraction", "Own training objective"),
        ("action_feasible_fraction", "Action constraint"),
        ("state_feasible_fraction", "Pitch + joint state constraints"),
        ("all_constraints_feasible_fraction", "All three constraints"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        for method in METHOD_ORDER:
            group = policy_frame[policy_frame["method"] == method].sort_values("training_seed")
            axis.plot(
                group["training_seed"],
                100.0 * group[metric],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=1.7,
                label=method,
            )
        axis.set_title(title)
        axis.set_ylim(-3, 103)
        axis.set_ylabel("Feasible episodes (%)")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Training seed")
    axes[-1, 1].set_xlabel("Training seed")
    axes[0, 0].legend()
    figure.suptitle("Symmetric Feasibility Comparison", y=1.01)
    save_figure(
        figure,
        figure_directory / "feasibility_by_training_seed.png",
        manifest,
        "Every policy is evaluated under action, pitch, joint-speed, and combined feasibility definitions.",
    )


def boxplot_with_seed_means(axis, episode_frame, policy_frame, metric, ylabel, title):
    methods = [method for method in METHOD_ORDER if method in set(episode_frame["method"])]
    values = [episode_frame.loc[episode_frame["method"] == method, metric] for method in methods]
    box = axis.boxplot(values, positions=method_positions(methods), widths=0.55, patch_artist=True)
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.25)
    for index, method in enumerate(methods):
        group = policy_frame[policy_frame["method"] == method]
        for _, row in group.iterrows():
            axis.scatter(
                index,
                row[f"{metric}_mean"],
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.4,
                marker=SEED_MARKERS.get(int(row["training_seed"]), "o"),
                s=52,
                zorder=4,
            )
    axis.set_xticks(method_positions(methods), methods)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)


def plot_pitch_distribution(episode_frame, policy_frame, figure_directory, manifest):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    boxplot_with_seed_means(
        axes[0],
        episode_frame,
        policy_frame,
        "pitch_abs_max_deg",
        "Episode maximum |pitch| (degrees)",
        "Raw Torso Pitch",
    )
    axes[0].axhline(90.0, color="black", linestyle="--", linewidth=1.2, label="90° activation")
    axes[0].axhline(99.0, color="#B22222", linestyle=":", linewidth=1.3, label="99° feasibility boundary")
    axes[0].legend()
    boxplot_with_seed_means(
        axes[1],
        episode_frame,
        policy_frame,
        "pitch_max_utilization",
        "Maximum pitch / 90° threshold",
        "Pitch Threshold Utilization",
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Cost activation")
    axes[1].axhline(1.1, color="#B22222", linestyle=":", linewidth=1.3, label="Feasibility boundary")
    axes[1].legend()
    figure.suptitle("Torso-Pitch Distributions Across All Evaluation Episodes", y=1.02)
    save_figure(
        figure,
        figure_directory / "pitch_episode_max_distributions.png",
        manifest,
        "Episode-maximum raw pitch and normalized threshold utilization; training-seed means are overlaid.",
    )


def plot_joint_distributions(episode_frame, policy_frame, figure_directory, manifest):
    figure, axes = plt.subplots(3, 2, figsize=(14, 13))
    for axis, name, threshold in zip(axes.flat, JOINT_NAMES, JOINT_SPEED_THRESHOLDS):
        metric = f"{name}_max_utilization"
        boxplot_with_seed_means(
            axis,
            episode_frame,
            policy_frame,
            metric,
            "Maximum speed / threshold",
            f"{name.replace('_', ' ').title()} (threshold {threshold:.2f} rad/s)",
        )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
        axis.axhline(1.1, color="#B22222", linestyle=":", linewidth=1.2)
    figure.suptitle("Six Joint-Speed Threshold-Utilization Distributions", y=1.01)
    save_figure(
        figure,
        figure_directory / "six_joint_speed_utilization_distributions.png",
        manifest,
        "Per-episode maximum speed divided by each joint's P99 threshold, with seed means overlaid.",
    )


def plot_action_comparison(episode_frame, policy_frame, figure_directory, manifest):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    boxplot_with_seed_means(
        axes[0],
        episode_frame,
        policy_frame,
        "maximum_absolute_action",
        "Episode maximum |action|",
        "Raw Maximum Action",
    )
    axes[0].axhline(float(ACTION_TORQUE_THRESHOLD), color="black", linestyle="--", linewidth=1.2, label="Cost activation")
    axes[0].axhline(float(ACTION_TORQUE_THRESHOLD + PERSISTENT_EPS), color="#B22222", linestyle=":", linewidth=1.3, label="Feasibility boundary")
    axes[0].legend()
    for method in METHOD_ORDER:
        group = policy_frame[policy_frame["method"] == method].sort_values("training_seed")
        axes[1].plot(
            group["training_seed"],
            100.0 * group["action_feasible_fraction"],
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=1.7,
            label=method,
        )
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Action-feasible episodes (%)")
    axes[1].set_title("Shilpa Action Constraint Applied to Every Policy")
    axes[1].set_ylim(-3, 103)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Symmetric Action Diagnostics", y=1.02)
    save_figure(
        figure,
        figure_directory / "action_metrics_all_policies.png",
        manifest,
        "Raw action distribution and Shilpa action-cost feasibility for all three policy families.",
    )


def plot_reward_safety(policy_frame, figure_directory, manifest):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    panels = (
        ("state_feasible_fraction", "Pitch + joint feasible episodes (%)"),
        ("all_constraints_feasible_fraction", "All-constraint feasible episodes (%)"),
    )
    for axis, (metric, label) in zip(axes, panels):
        for method in METHOD_ORDER:
            group = policy_frame[policy_frame["method"] == method]
            axis.scatter(
                100.0 * group[metric],
                group["reward_mean"],
                color=METHOD_COLORS[method],
                s=75,
                label=method,
            )
            for _, row in group.iterrows():
                axis.annotate(
                    str(int(row["training_seed"])),
                    (100.0 * row[metric], row["reward_mean"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.set_xlabel(label)
        axis.set_ylabel("Mean reward")
        axis.set_xlim(-3, 103)
        axis.grid(alpha=0.25)
    axes[0].set_title("Reward vs State Safety")
    axes[1].set_title("Reward vs Action + State Safety")
    axes[0].legend()
    figure.suptitle("Reward–Safety Trade-offs by Trained Policy", y=1.02)
    save_figure(
        figure,
        figure_directory / "reward_vs_common_safety_metrics.png",
        manifest,
        "Each point is one trained seed; labels identify training seeds and higher/right is better.",
    )


def plot_method_summary(method_frame, figure_directory, manifest):
    panels = (
        ("reward_mean", "Reward", 1.0),
        ("action_feasible_fraction", "Action feasibility (%)", 100.0),
        ("state_feasible_fraction", "State feasibility (%)", 100.0),
        ("all_constraints_feasible_fraction", "All-constraint feasibility (%)", 100.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    positions = method_positions(list(method_frame["method"]))
    for axis, (metric, title, scale) in zip(axes.flat, panels):
        means = scale * method_frame[f"{metric}_across_seed_mean"].to_numpy()
        lows = scale * method_frame[f"{metric}_across_seed_ci95_low"].to_numpy()
        highs = scale * method_frame[f"{metric}_across_seed_ci95_high"].to_numpy()
        if scale == 100.0:
            lows = np.clip(lows, 0.0, 100.0)
            highs = np.clip(highs, 0.0, 100.0)
        colors = [METHOD_COLORS[method] for method in method_frame["method"]]
        axis.bar(positions, means, color=colors, alpha=0.75)
        axis.errorbar(
            positions,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            color="black",
            capsize=4,
        )
        axis.set_xticks(positions, method_frame["method"], rotation=12)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if scale == 100.0:
            axis.set_ylim(0.0, 105.0)
    figure.suptitle("Algorithm Means Across Training Seeds (95% t Intervals)", y=1.01)
    save_figure(
        figure,
        figure_directory / "method_means_across_training_seeds.png",
        manifest,
        "Mean policy results across independent training seeds; intervals use training seeds, not evaluation episodes.",
    )


def plot_matched_timeseries(selected_frame, figure_directory, manifest, plot_episode):
    thresholds = np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64)
    for training_seed in sorted(selected_frame["training_seed"].unique()):
        seed_frame = selected_frame[selected_frame["training_seed"] == training_seed]
        figure, axes = plt.subplots(7, 1, figsize=(13, 18), sharex=True)
        for method in METHOD_ORDER:
            group = seed_frame[seed_frame["method"] == method].sort_values("timestep")
            if group.empty:
                continue
            axes[0].plot(
                group["time_seconds"],
                np.degrees(group["torso_pitch"]),
                color=METHOD_COLORS[method],
                linewidth=1.25,
                label=method,
            )
            for index, name in enumerate(JOINT_NAMES, start=1):
                axes[index].plot(
                    group["time_seconds"],
                    group[f"{name}_speed"],
                    color=METHOD_COLORS[method],
                    linewidth=1.1,
                    label=method,
                )
        axes[0].axhline(
            90.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Cost activation (±90°)",
        )
        axes[0].axhline(-90.0, color="black", linestyle="--", linewidth=1.0)
        axes[0].axhline(
            99.0,
            color="#B22222",
            linestyle=":",
            linewidth=1.0,
            label="Feasibility boundary (±99°)",
        )
        axes[0].axhline(-99.0, color="#B22222", linestyle=":", linewidth=1.0)
        axes[0].set_ylabel("Pitch (deg)")
        axes[0].set_title("Torso pitch")
        for index, (name, threshold) in enumerate(zip(JOINT_NAMES, thresholds), start=1):
            axes[index].axhline(threshold, color="black", linestyle="--", linewidth=1.0)
            axes[index].axhline(
                1.1 * threshold,
                color="#B22222",
                linestyle=":",
                linewidth=1.0,
            )
            axes[index].set_ylabel("rad/s")
            axes[index].set_title(name.replace("_", " ").title())
        for axis in axes:
            axis.grid(alpha=0.22)
        axes[0].legend(ncol=3)
        axes[-1].set_xlabel("Time (s)")
        evaluation_seed = int(seed_frame["evaluation_seed"].iloc[0])
        figure.suptitle(
            f"Matched Final Inference: Training Seed {training_seed}, "
            f"Evaluation Episode {plot_episode} (Reset {evaluation_seed})",
            y=1.005,
        )
        filename = f"matched_timeseries_training_seed_{training_seed}.png"
        save_figure(
            figure,
            figure_directory / filename,
            manifest,
            "Raw pitch and six joint speeds for the same reset across all three methods.",
        )


def write_results_readme(output_directory, method_frame, seed_tests, pareto_frame, args):
    lines = [
        "# Unified RCAC HalfCheetah Evaluation",
        "",
        "All policies were evaluated deterministically on the same reset seeds. Every policy was measured using Shilpa's action constraint, the pitch constraint, and all six joint-speed constraints regardless of its training objective.",
        "",
        "## Method means across training seeds",
        "",
        "| Method | Reward | Action feasible | State feasible | All feasible |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in method_frame.iterrows():
        lines.append(
            f"| {row['method']} | {row['reward_mean_across_seed_mean']:.1f} | "
            f"{100.0 * row['action_feasible_fraction_across_seed_mean']:.1f}% | "
            f"{100.0 * row['state_feasible_fraction_across_seed_mean']:.1f}% | "
            f"{100.0 * row['all_constraints_feasible_fraction_across_seed_mean']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Reward and feasibility are separate outcomes; a method is not universally better unless it is no worse on both.",
            "- Raw action-cost, pitch-cost, and joint-cost magnitudes are not directly compared because their scaling differs.",
            "- Algorithm-level claims use independent training seeds. The 100 matched resets characterize each selected policy but are not 100 independent training runs.",
            f"- With {len(args.seeds)} training seeds, the exact two-sided sign-flip test cannot reach p < 0.05 when only four nonzero paired differences are available; confidence intervals and seed consistency are therefore emphasized.",
            "- `pareto_by_seed.csv` identifies methods that are not dominated simultaneously on reward and safety.",
            "",
            "## Shilpa OG cost",
            "",
            "Shilpa's original cost is `max(max(abs(action)) - 0.5, 0)`. It controls peak action magnitude, an upstream control-effort proxy. It does not directly measure torso pitch or joint speed, but limiting actions can indirectly reduce both.",
            "",
            "## Primary evidence files",
            "",
            "- `episode_metrics.csv`: one row per policy and evaluation episode.",
            "- `policy_summary.csv`: one row per selected trained policy.",
            "- `method_summary_across_training_seeds.csv`: algorithm means across independent seeds.",
            "- `seed_level_pairwise_tests.csv`: paired training-seed comparisons.",
            "- `within_seed_paired_reset_tests.csv`: matched-reset comparisons for fixed selected policies.",
            "- `pareto_by_seed.csv`: reward-safety dominance results.",
            "- `selected_episode_timestep_metrics.csv`: raw traces used in matched time-series figures.",
        ]
    )
    significant = seed_tests[seed_tests["exact_sign_flip_pvalue"] < 0.05]
    lines.extend(["", "## Automatic statistical caution", ""])
    if significant.empty:
        lines.append(
            "No seed-level pairwise comparison reached p < 0.05 under the exact paired sign-flip test. Treat method advantages as descriptive unless more independent training seeds are added."
        )
    else:
        lines.append(
            "At least one exact paired seed-level comparison reached p < 0.05; inspect `seed_level_pairwise_tests.csv` and confirm that the result is meaningful across reward and safety rather than a single isolated metric."
        )
    nondominated_counts = (
        pareto_frame[pareto_frame["pareto_nondominated"]]
        .groupby(["comparison", "method"])
        .size()
        .reset_index(name="nondominated_seed_count")
    )
    lines.extend(["", "## Pareto summary", ""])
    for _, row in nondominated_counts.iterrows():
        lines.append(
            f"- {row['comparison']}: {row['method']} was nondominated in "
            f"{int(row['nondominated_seed_count'])}/{len(args.seeds)} seeds."
        )
    (output_directory / "README_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path, frame):
    frame.to_csv(path, index=False)


def create_outputs(
    output_directory,
    episode_frame,
    policy_frame,
    method_frame,
    selected_frame,
    seed_tests,
    within_seed_tests,
    pareto_frame,
    args,
):
    write_csv(output_directory / "episode_metrics.csv", episode_frame)
    write_csv(output_directory / "policy_summary.csv", policy_frame)
    write_csv(output_directory / "method_summary_across_training_seeds.csv", method_frame)
    write_csv(output_directory / "selected_episode_timestep_metrics.csv", selected_frame)
    write_csv(output_directory / "seed_level_pairwise_tests.csv", seed_tests)
    write_csv(output_directory / "within_seed_paired_reset_tests.csv", within_seed_tests)
    write_csv(output_directory / "pareto_by_seed.csv", pareto_frame)
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=False)
    manifest = []
    plot_reward_by_seed(policy_frame, figure_directory, manifest)
    plot_feasibility_by_seed(policy_frame, figure_directory, manifest)
    plot_pitch_distribution(episode_frame, policy_frame, figure_directory, manifest)
    plot_joint_distributions(episode_frame, policy_frame, figure_directory, manifest)
    plot_action_comparison(episode_frame, policy_frame, figure_directory, manifest)
    plot_reward_safety(policy_frame, figure_directory, manifest)
    plot_method_summary(method_frame, figure_directory, manifest)
    plot_matched_timeseries(selected_frame, figure_directory, manifest, args.plot_episode)
    pd.DataFrame(manifest).to_csv(figure_directory / "figure_manifest.csv", index=False)
    write_results_readme(output_directory, method_frame, seed_tests, pareto_frame, args)


def all_timestep_fieldnames():
    fields = [
        "method",
        "training_objective",
        "training_seed",
        "run",
        "episode",
        "evaluation_seed",
        "timestep",
        "time_seconds",
        "reward",
        "maximum_absolute_action",
        "action_cost",
        "torso_pitch_raw",
        "torso_pitch",
        "torso_pitch_abs",
        "pitch_cost",
        "joint_speed_cost",
        "combined_state_cost",
    ]
    for index, name in enumerate(JOINT_NAMES, start=1):
        fields.extend(
            [
                f"action_{index}",
                f"{name}_velocity",
                f"{name}_speed",
                f"{name}_cost",
            ]
        )
    return fields


def main():
    args = parse_arguments()
    validate_arguments(args)
    repo_root = Path(__file__).resolve().parent
    specs = build_policy_specs(repo_root, args)
    for spec in specs:
        print(
            f"{spec.method}: seed={spec.training_seed} run={spec.run} "
            f"checkpoint={spec.actor_path}"
        )
    validate_policy_specs(specs)
    if args.dry_run:
        print("All checkpoints and available metadata passed validation.")
        return
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for evaluation. Run this script from the same "
            "Python environment used for RCAC training."
        ) from TORCH_IMPORT_ERROR
    if HalfCheetahWithPos is None:
        raise RuntimeError(
            "The HalfCheetah environment dependencies are unavailable. Run this "
            "script from the same Python environment used for RCAC training."
        ) from ENV_IMPORT_ERROR
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_directory = output_root / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_directory.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    np.random.seed(args.analysis_seed)
    torch.manual_seed(args.analysis_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.analysis_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    selected_timestep_rows = []
    all_episode_rows = []
    timestep_handle = None
    timestep_writer = None
    if args.save_all_timesteps:
        timestep_handle = (output_directory / "all_timestep_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        )
        timestep_writer = csv.DictWriter(timestep_handle, fieldnames=all_timestep_fieldnames())
        timestep_writer.writeheader()
    try:
        for index, spec in enumerate(specs, start=1):
            print(
                f"Evaluating policy {index}/{len(specs)}: {spec.method}, "
                f"training seed {spec.training_seed}"
            )
            all_episode_rows.extend(
                evaluate_policy(
                    spec,
                    args,
                    device,
                    selected_timestep_rows,
                    timestep_writer,
                )
            )
    finally:
        if timestep_handle is not None:
            timestep_handle.close()
    episode_frame = pd.DataFrame(all_episode_rows)
    selected_frame = pd.DataFrame(selected_timestep_rows)
    expected_episode_rows = len(specs) * args.episodes
    if len(episode_frame) != expected_episode_rows:
        raise RuntimeError(
            f"Expected {expected_episode_rows} episode rows, produced {len(episode_frame)}"
        )
    policy_frame = summarize_policies(episode_frame)
    method_frame = summarize_methods(policy_frame)
    seed_tests = seed_level_pairwise_tests(policy_frame, args)
    within_seed_tests = within_seed_pairwise_tests(episode_frame)
    pareto_frame = pareto_rows(policy_frame)
    create_outputs(
        output_directory,
        episode_frame,
        policy_frame,
        method_frame,
        selected_frame,
        seed_tests,
        within_seed_tests,
        pareto_frame,
        args,
    )
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repository_root": str(repo_root),
        "output_directory": str(output_directory),
        "methods": list(METHOD_ORDER),
        "training_seeds": args.seeds,
        "action_runs": args.action_runs,
        "scalar_runs": args.scalar_runs,
        "multi_runs": args.multi_runs,
        "episodes_per_policy": args.episodes,
        "evaluation_seed_start": args.base_seed,
        "evaluation_seed_end": args.base_seed + args.episodes - 1,
        "plot_episode": args.plot_episode,
        "plot_evaluation_seed": args.base_seed + args.plot_episode - 1,
        "deterministic_mean_actions": True,
        "persistent_epsilon": args.persistent_eps,
        "action_torque_threshold": float(ACTION_TORQUE_THRESHOLD),
        "pitch_threshold_rad": float(PITCH_THRESHOLD),
        "pitch_threshold_deg": float(np.degrees(PITCH_THRESHOLD)),
        "joint_names": list(JOINT_NAMES),
        "joint_speed_thresholds": np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64).tolist(),
        "activation": args.activation,
        "device": str(device),
        "bootstrap_samples": args.bootstrap_samples,
        "save_all_timesteps": args.save_all_timesteps,
        "policies": [
            {
                "method": spec.method,
                "training_objective": spec.training_objective,
                "training_seed": spec.training_seed,
                "run": spec.run,
                "actor_path": str(spec.actor_path),
                "metadata_path": str(spec.metadata_path) if spec.metadata_path else None,
            }
            for spec in specs
        ],
    }
    (output_directory / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Outputs: {output_directory}")
    print(method_frame.to_string(index=False))


if __name__ == "__main__":
    main()
