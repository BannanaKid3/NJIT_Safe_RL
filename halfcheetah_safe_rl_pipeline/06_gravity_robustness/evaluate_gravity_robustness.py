import argparse
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
import torch
import torch.nn as nn
from scipy.stats import t as student_t

from configs.halfcheetah_combined_safety import (
    JOINT_NAMES,
    JOINT_SPEED_THRESHOLDS,
    PERSISTENT_EPS,
    PITCH_THRESHOLD,
)
from envs.half_cheetah import ACTION_TORQUE_THRESHOLD
from envs.half_cheetah_gravity_perturbed_costs import (
    HalfCheetahCombinedMaxCostGravityPerturbed,
)
from gravity_robustness_common import (
    MULTI_BASE_VERSION,
    SCALAR_BASE_VERSION,
    multi_gravity_version,
    parse_float_list,
    parse_int_list,
    scalar_gravity_version,
)


METHODS = ("Scalar combined", "Multi-constraint")
METHOD_DIRECTORIES = {
    "Scalar combined": "HalfCheetahCombinedMaxCost",
    "Multi-constraint": "HalfCheetahMultiConstraintCost",
}
TRAIN_SIGMA_COLORS = {0.0: "#6B7280", 0.5: "#2F6BFF", 0.7: "#F28E2B"}


@dataclass(frozen=True)
class PolicySpec:
    method: str
    train_sigma: float
    training_seed: int
    run: int
    version: str
    actor_path: Path

    @property
    def condition(self):
        if self.train_sigma == 0.0:
            return f"{self.method} nominal"
        return f"{self.method} train σ={self.train_sigma:g}"


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sigmas", type=parse_float_list, default=[0.5, 0.7])
    parser.add_argument("--eval_sigmas", type=parse_float_list, default=[0.0, 1.0, 2.0])
    parser.add_argument("--rho", type=float, default=0.001)
    parser.add_argument("--seeds", type=parse_int_list, default=[5, 11, 17, 23])
    parser.add_argument("--robust_runs", type=parse_int_list, default=[1, 2, 3, 4])
    parser.add_argument("--nominal_scalar_runs", type=parse_int_list, default=[4, 5, 6, 7])
    parser.add_argument("--nominal_multi_runs", type=parse_int_list, default=[4, 5, 6, 7])
    parser.add_argument("--exclude_nominal", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=2301)
    parser.add_argument("--plot_episode", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--persistent_eps", type=float, default=float(PERSISTENT_EPS))
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--output_root",
        default="analysis/outputs/halfcheetah_gravity_robustness",
    )
    return parser.parse_args()


def validate_arguments(args):
    if len(args.seeds) != len(args.robust_runs):
        raise ValueError("seeds and robust_runs must have equal lengths")
    if not args.exclude_nominal:
        if len(args.seeds) != len(args.nominal_scalar_runs):
            raise ValueError("seeds and nominal_scalar_runs must have equal lengths")
        if len(args.seeds) != len(args.nominal_multi_runs):
            raise ValueError("seeds and nominal_multi_runs must have equal lengths")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("training seeds must be unique")
    if len(set(args.train_sigmas)) != len(args.train_sigmas):
        raise ValueError("train_sigmas must be unique")
    if len(set(args.eval_sigmas)) != len(args.eval_sigmas):
        raise ValueError("eval_sigmas must be unique")
    if any(value < 0.0 for value in args.train_sigmas + args.eval_sigmas):
        raise ValueError("gravity sigmas cannot be negative")
    if args.rho < 0.0:
        raise ValueError("rho cannot be negative")
    if args.episodes < 1 or args.max_episode_steps < 1:
        raise ValueError("episode counts must be positive")
    if not 1 <= args.plot_episode <= args.episodes:
        raise ValueError("plot_episode must lie within evaluated episodes")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def latest_checkpoint(directory):
    candidates = []
    pattern = re.compile(r"^Best_RCAC_actor_(\d+)$")
    for path in directory.glob("Best_RCAC_actor_*"):
        match = pattern.match(path.name)
        if path.is_file() and match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return directory / "Best_RCAC_actor"


def policy_directory(repo_root, method, version, run):
    return repo_root / "models" / METHOD_DIRECTORIES[method] / version / f"run{run}"


def build_policy_specs(repo_root, args):
    specs = []
    if not args.exclude_nominal:
        for method, version, runs in (
            ("Scalar combined", SCALAR_BASE_VERSION, args.nominal_scalar_runs),
            ("Multi-constraint", MULTI_BASE_VERSION, args.nominal_multi_runs),
        ):
            for seed, run in zip(args.seeds, runs):
                directory = policy_directory(repo_root, method, version, run)
                specs.append(
                    PolicySpec(method, 0.0, seed, run, version, latest_checkpoint(directory))
                )
    for sigma in args.train_sigmas:
        versions = {
            "Scalar combined": scalar_gravity_version(sigma, args.rho),
            "Multi-constraint": multi_gravity_version(sigma, args.rho),
        }
        for method in METHODS:
            for seed, run in zip(args.seeds, args.robust_runs):
                version = versions[method]
                directory = policy_directory(repo_root, method, version, run)
                specs.append(
                    PolicySpec(method, float(sigma), seed, run, version, latest_checkpoint(directory))
                )
    return specs


def validate_policy_specs(specs):
    missing = [spec.actor_path for spec in specs if not spec.actor_path.exists()]
    if missing:
        raise FileNotFoundError("Missing actor checkpoints:\n" + "\n".join(str(path) for path in missing))


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
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def load_actor(spec, environment, activation, device):
    state = clean_state_dict(torch_load(spec.actor_path, device))
    required = {"fc1.weight", "fc2.weight", "fc3.weight", "mean_layer.weight"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"Unexpected actor format in {spec.actor_path}: missing {sorted(missing)}")
    state_dim = int(state["fc1.weight"].shape[1])
    hidden_width = int(state["fc1.weight"].shape[0])
    action_dim = int(state["mean_layer.weight"].shape[0])
    expected_state = int(environment.observation_space.shape[0])
    expected_action = int(environment.action_space.shape[0])
    if (state_dim, action_dim) != (expected_state, expected_action):
        raise ValueError(
            f"{spec.actor_path} dimensions {(state_dim, action_dim)} do not match "
            f"environment {(expected_state, expected_action)}"
        )
    max_action = float(np.asarray(environment.action_space.high).reshape(-1)[0])
    actor = ActorGaussian(state_dim, hidden_width, action_dim, max_action, activation)
    actor.load_state_dict(state, strict=True)
    actor.to(device)
    actor.eval()
    return actor


def normalizer_path(actor_path):
    return actor_path.with_name(actor_path.name.replace("_actor", "_state_norm", 1))


def load_normalizer(actor_path):
    path = normalizer_path(actor_path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return CompatibleUnpickler(handle).load()


def reset_observation(environment, seed):
    result = environment.reset(seed=seed)
    while isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float32)


def policy_action(actor, observation, normalizer, device):
    actor_input = observation if normalizer is None else normalizer(observation, update=False)
    tensor = torch.as_tensor(actor_input, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return actor(tensor).cpu().numpy().reshape(-1)


def episode_row(spec, eval_sigma, episode, evaluation_seed, trace, threshold):
    joint_utilizations = np.asarray(trace["joint_utilizations"], dtype=np.float64)
    row = {
        "method": spec.method,
        "condition": spec.condition,
        "train_sigma": spec.train_sigma,
        "eval_sigma": float(eval_sigma),
        "training_seed": spec.training_seed,
        "run": spec.run,
        "version": spec.version,
        "actor_path": str(spec.actor_path),
        "episode": episode,
        "evaluation_seed": evaluation_seed,
        "steps": len(trace["rewards"]),
        "reward": float(np.sum(trace["rewards"])),
        "combined_cost_max": float(np.max(trace["combined_costs"])),
        "combined_cost_total": float(np.sum(trace["combined_costs"])),
        "pitch_cost_max": float(np.max(trace["pitch_costs"])),
        "pitch_cost_total": float(np.sum(trace["pitch_costs"])),
        "joint_cost_max": float(np.max(trace["joint_costs"])),
        "joint_cost_total": float(np.sum(trace["joint_costs"])),
        "pitch_feasible": bool(np.max(trace["pitch_costs"]) <= threshold),
        "joint_feasible": bool(np.max(trace["joint_costs"]) <= threshold),
        "state_feasible": bool(np.max(trace["combined_costs"]) <= threshold),
        "pitch_abs_max_deg": float(np.degrees(np.max(trace["pitch_abs"]))),
        "joint_max_utilization": float(np.max(joint_utilizations)),
        "maximum_absolute_action": float(np.max(trace["max_actions"])),
        "action_cost_max": float(np.max(trace["action_costs"])),
        "action_feasible": bool(np.max(trace["action_costs"]) <= threshold),
        "gravity_noise_mean": float(np.mean(trace["gravity_noise"])),
        "gravity_noise_sd": float(np.std(trace["gravity_noise"], ddof=1)),
        "gravity_noise_min": float(np.min(trace["gravity_noise"])),
        "gravity_noise_max": float(np.max(trace["gravity_noise"])),
    }
    for index, name in enumerate(JOINT_NAMES):
        row[f"{name}_max_utilization"] = float(np.max(joint_utilizations[:, index]))
    return row


def selected_timestep_row(spec, eval_sigma, episode, evaluation_seed, step, action, reward, info):
    row = {
        "method": spec.method,
        "condition": spec.condition,
        "train_sigma": spec.train_sigma,
        "eval_sigma": float(eval_sigma),
        "training_seed": spec.training_seed,
        "run": spec.run,
        "episode": episode,
        "evaluation_seed": evaluation_seed,
        "timestep": step,
        "time_seconds": step * 0.05,
        "reward": float(reward),
        "gravity_z": float(info["gravity_z"]),
        "gravity_noise": float(info["gravity_noise"]),
        "torso_pitch": float(info["torso_pitch"]),
        "pitch_cost": float(info["pitch_cost"]),
        "joint_speed_cost": float(info["joint_speed_cost"]),
        "combined_cost": float(info["combined_cost"]),
        "maximum_absolute_action": float(np.max(np.abs(action))),
    }
    joint_speeds = np.asarray(info["joint_speeds"], dtype=np.float64)
    for index, name in enumerate(JOINT_NAMES):
        row[f"action_{index + 1}"] = float(action[index])
        row[f"{name}_speed"] = float(joint_speeds[index])
    return row


def evaluate_policy(spec, eval_sigma, args, selected_rows):
    environment = HalfCheetahCombinedMaxCostGravityPerturbed(sigma_gravity=eval_sigma)
    actor = load_actor(spec, environment, args.activation, args.device)
    normalizer = load_normalizer(spec.actor_path)
    rows = []
    thresholds = np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64)
    for episode in range(1, args.episodes + 1):
        evaluation_seed = args.base_seed + episode - 1
        observation = reset_observation(environment, evaluation_seed)
        trace = {
            "rewards": [],
            "combined_costs": [],
            "pitch_costs": [],
            "joint_costs": [],
            "pitch_abs": [],
            "joint_utilizations": [],
            "max_actions": [],
            "action_costs": [],
            "gravity_noise": [],
        }
        for step in range(1, args.max_episode_steps + 1):
            action = policy_action(actor, observation, normalizer, args.device)
            observation, reward, _, truncated, terminated, info = environment.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            max_action = float(np.max(np.abs(action)))
            action_cost = max(max_action - float(ACTION_TORQUE_THRESHOLD), 0.0)
            trace["rewards"].append(float(reward))
            trace["combined_costs"].append(float(info["combined_cost"]))
            trace["pitch_costs"].append(float(info["pitch_cost"]))
            trace["joint_costs"].append(float(info["joint_speed_cost"]))
            trace["pitch_abs"].append(float(info["torso_pitch_abs"]))
            trace["joint_utilizations"].append(np.asarray(info["joint_speeds"]) / thresholds)
            trace["max_actions"].append(max_action)
            trace["action_costs"].append(action_cost)
            trace["gravity_noise"].append(float(info["gravity_noise"]))
            if episode == args.plot_episode:
                selected_rows.append(
                    selected_timestep_row(
                        spec,
                        eval_sigma,
                        episode,
                        evaluation_seed,
                        step,
                        action,
                        reward,
                        info,
                    )
                )
            if truncated or terminated:
                break
        rows.append(episode_row(spec, eval_sigma, episode, evaluation_seed, trace, args.persistent_eps))
    environment.close()
    return rows


def t_interval(values):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    mean = float(np.mean(values))
    if count < 2:
        return mean, math.nan, math.nan, 0.0
    standard_deviation = float(np.std(values, ddof=1))
    margin = float(student_t.ppf(0.975, count - 1) * standard_deviation / math.sqrt(count))
    return mean, mean - margin, mean + margin, standard_deviation


def wilson_interval(successes, count, z=1.959963984540054):
    if count == 0:
        return math.nan, math.nan
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)) / denominator
    return center - margin, center + margin


def summarize_policies(episode_frame):
    rows = []
    keys = ["method", "condition", "train_sigma", "eval_sigma", "training_seed", "run", "version", "actor_path"]
    for values, group in episode_frame.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        reward_mean, reward_low, reward_high, reward_sd = t_interval(group["reward"])
        row.update(
            {
                "episodes": len(group),
                "reward_mean": reward_mean,
                "reward_sd": reward_sd,
                "reward_ci95_low": reward_low,
                "reward_ci95_high": reward_high,
                "combined_cost_max_mean": float(group["combined_cost_max"].mean()),
                "combined_cost_max_worst": float(group["combined_cost_max"].max()),
                "pitch_cost_max_mean": float(group["pitch_cost_max"].mean()),
                "joint_cost_max_mean": float(group["joint_cost_max"].mean()),
                "pitch_abs_max_deg_mean": float(group["pitch_abs_max_deg"].mean()),
                "joint_max_utilization_mean": float(group["joint_max_utilization"].mean()),
                "maximum_absolute_action_mean": float(group["maximum_absolute_action"].mean()),
            }
        )
        for metric in ("pitch_feasible", "joint_feasible", "state_feasible", "action_feasible"):
            successes = int(group[metric].sum())
            low, high = wilson_interval(successes, len(group))
            row[f"{metric}_fraction"] = successes / len(group)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_methods(policy_frame):
    metrics = (
        "reward_mean",
        "combined_cost_max_mean",
        "pitch_cost_max_mean",
        "joint_cost_max_mean",
        "pitch_abs_max_deg_mean",
        "joint_max_utilization_mean",
        "maximum_absolute_action_mean",
        "pitch_feasible_fraction",
        "joint_feasible_fraction",
        "state_feasible_fraction",
        "action_feasible_fraction",
    )
    rows = []
    keys = ["method", "condition", "train_sigma", "eval_sigma"]
    for values, group in policy_frame.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row["training_seeds"] = len(group)
        row["seed_list"] = ",".join(str(int(value)) for value in sorted(group["training_seed"]))
        for metric in metrics:
            mean, low, high, standard_deviation = t_interval(group[metric])
            row[f"{metric}_across_seed_mean"] = mean
            row[f"{metric}_across_seed_sd"] = standard_deviation
            row[f"{metric}_across_seed_ci95_low"] = low
            row[f"{metric}_across_seed_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(differences):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.abs(differences) > 1e-15]
    count = len(differences)
    if count == 0:
        return 1.0
    observed = abs(float(np.mean(differences)))
    extreme = 0
    total = 2 ** count
    for signs in itertools.product((-1.0, 1.0), repeat=count):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        if statistic >= observed - 1e-15:
            extreme += 1
    return extreme / total


def paired_training_tests(policy_frame):
    rows = []
    metrics = {
        "reward_mean": "higher",
        "state_feasible_fraction": "higher",
        "combined_cost_max_mean": "lower",
        "pitch_feasible_fraction": "higher",
        "joint_feasible_fraction": "higher",
    }
    for method in METHODS:
        method_frame = policy_frame[policy_frame["method"] == method]
        for eval_sigma in sorted(method_frame["eval_sigma"].unique()):
            sigma_frame = method_frame[method_frame["eval_sigma"] == eval_sigma]
            train_sigmas = sorted(sigma_frame["train_sigma"].unique())
            for sigma_a, sigma_b in itertools.combinations(train_sigmas, 2):
                first = sigma_frame[sigma_frame["train_sigma"] == sigma_a]
                second = sigma_frame[sigma_frame["train_sigma"] == sigma_b]
                merged = first.merge(second, on="training_seed", suffixes=("_a", "_b"))
                for metric, direction in metrics.items():
                    differences = merged[f"{metric}_a"].to_numpy() - merged[f"{metric}_b"].to_numpy()
                    mean, low, high, standard_deviation = t_interval(differences)
                    rows.append(
                        {
                            "method": method,
                            "eval_sigma": eval_sigma,
                            "train_sigma_a": sigma_a,
                            "train_sigma_b": sigma_b,
                            "metric": metric,
                            "preferred_direction": direction,
                            "matched_training_seeds": len(merged),
                            "mean_difference_a_minus_b": mean,
                            "difference_sd": standard_deviation,
                            "ci95_low": low,
                            "ci95_high": high,
                            "exact_sign_flip_pvalue": exact_sign_flip_pvalue(differences),
                        }
                    )
    return pd.DataFrame(rows)


def paired_method_tests(policy_frame):
    rows = []
    metrics = {
        "reward_mean": "higher",
        "state_feasible_fraction": "higher",
        "combined_cost_max_mean": "lower",
        "pitch_feasible_fraction": "higher",
        "joint_feasible_fraction": "higher",
    }
    for train_sigma in sorted(policy_frame["train_sigma"].unique()):
        for eval_sigma in sorted(policy_frame["eval_sigma"].unique()):
            frame = policy_frame[
                (policy_frame["train_sigma"] == train_sigma)
                & (policy_frame["eval_sigma"] == eval_sigma)
            ]
            first = frame[frame["method"] == METHODS[0]]
            second = frame[frame["method"] == METHODS[1]]
            merged = first.merge(second, on="training_seed", suffixes=("_scalar", "_multi"))
            for metric, direction in metrics.items():
                differences = (
                    merged[f"{metric}_scalar"].to_numpy()
                    - merged[f"{metric}_multi"].to_numpy()
                )
                mean, low, high, standard_deviation = t_interval(differences)
                rows.append(
                    {
                        "train_sigma": train_sigma,
                        "eval_sigma": eval_sigma,
                        "method_a": METHODS[0],
                        "method_b": METHODS[1],
                        "metric": metric,
                        "preferred_direction": direction,
                        "matched_training_seeds": len(merged),
                        "mean_difference_scalar_minus_multi": mean,
                        "difference_sd": standard_deviation,
                        "ci95_low": low,
                        "ci95_high": high,
                        "exact_sign_flip_pvalue": exact_sign_flip_pvalue(differences),
                    }
                )
    return pd.DataFrame(rows)


def exact_mcnemar_pvalue(a_only, b_only):
    discordant = int(a_only + b_only)
    if discordant == 0:
        return 1.0
    smaller = min(int(a_only), int(b_only))
    probability = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * probability / (2 ** discordant))


def matched_reset_tests(episode_frame):
    rows = []
    for method in METHODS:
        method_frame = episode_frame[episode_frame["method"] == method]
        for eval_sigma in sorted(method_frame["eval_sigma"].unique()):
            for training_seed in sorted(method_frame["training_seed"].unique()):
                frame = method_frame[
                    (method_frame["eval_sigma"] == eval_sigma)
                    & (method_frame["training_seed"] == training_seed)
                ]
                train_sigmas = sorted(frame["train_sigma"].unique())
                for sigma_a, sigma_b in itertools.combinations(train_sigmas, 2):
                    first = frame[frame["train_sigma"] == sigma_a]
                    second = frame[frame["train_sigma"] == sigma_b]
                    merged = first.merge(second, on="evaluation_seed", suffixes=("_a", "_b"))
                    for metric in ("reward", "combined_cost_max"):
                        differences = merged[f"{metric}_a"].to_numpy() - merged[f"{metric}_b"].to_numpy()
                        mean, low, high, standard_deviation = t_interval(differences)
                        rows.append(
                            {
                                "method": method,
                                "eval_sigma": eval_sigma,
                                "training_seed": training_seed,
                                "train_sigma_a": sigma_a,
                                "train_sigma_b": sigma_b,
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
                    first_safe = merged["state_feasible_a"].astype(bool)
                    second_safe = merged["state_feasible_b"].astype(bool)
                    a_only = int((first_safe & ~second_safe).sum())
                    b_only = int((~first_safe & second_safe).sum())
                    rows.append(
                        {
                            "method": method,
                            "eval_sigma": eval_sigma,
                            "training_seed": training_seed,
                            "train_sigma_a": sigma_a,
                            "train_sigma_b": sigma_b,
                            "test_type": "matched_reset_mcnemar",
                            "metric": "state_feasible",
                            "matched_resets": len(merged),
                            "mean_difference_a_minus_b": float(first_safe.mean() - second_safe.mean()),
                            "difference_sd": math.nan,
                            "ci95_low": math.nan,
                            "ci95_high": math.nan,
                            "a_safe_b_unsafe": a_only,
                            "a_unsafe_b_safe": b_only,
                            "exact_mcnemar_pvalue": exact_mcnemar_pvalue(a_only, b_only),
                        }
                    )
    return pd.DataFrame(rows)


def retention_table(policy_frame):
    base = policy_frame[policy_frame["eval_sigma"] == 0.0].copy()
    if base.empty:
        return pd.DataFrame()
    keys = ["method", "condition", "train_sigma", "training_seed", "run"]
    base = base[keys + ["reward_mean", "state_feasible_fraction"]].rename(
        columns={
            "reward_mean": "reward_mean_at_sigma0",
            "state_feasible_fraction": "state_feasible_at_sigma0",
        }
    )
    merged = policy_frame.merge(base, on=keys, how="left")
    denominator = merged["reward_mean_at_sigma0"].where(
        merged["reward_mean_at_sigma0"].abs() > 1e-12
    )
    merged["reward_retention_fraction"] = merged["reward_mean"] / denominator
    merged["state_feasibility_change"] = merged["state_feasible_fraction"] - merged["state_feasible_at_sigma0"]
    return merged


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def sigma_color(sigma, index):
    return TRAIN_SIGMA_COLORS.get(float(sigma), plt.get_cmap("tab10")(index))


def sigma_label(sigma):
    return "Nominal training" if sigma == 0.0 else f"Train σ={sigma:g}"


def plot_robustness_metric(method_frame, metric, ylabel, title, path, threshold=None, percent=False):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True)
    for axis, method in zip(axes, METHODS):
        frame = method_frame[method_frame["method"] == method]
        for index, train_sigma in enumerate(sorted(frame["train_sigma"].unique())):
            group = frame[frame["train_sigma"] == train_sigma].sort_values("eval_sigma")
            scale = 100.0 if percent else 1.0
            mean = scale * group[f"{metric}_across_seed_mean"].to_numpy()
            standard_deviation = scale * group[f"{metric}_across_seed_sd"].to_numpy()
            x = group["eval_sigma"].to_numpy()
            color = sigma_color(train_sigma, index)
            axis.plot(x, mean, marker="o", color=color, linewidth=2.2, label=sigma_label(train_sigma))
            axis.fill_between(x, mean - standard_deviation, mean + standard_deviation, color=color, alpha=0.18)
        if threshold is not None:
            axis.axhline(threshold, color="black", linestyle="--", linewidth=1.3, label="Safety threshold")
        axis.set_title(method, fontweight="bold")
        axis.set_xlabel("Evaluation gravity σ (m/s²)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        if percent:
            axis.set_ylim(-3, 103)
    figure.suptitle(title, fontweight="bold")
    save_figure(figure, path)


def plot_constraint_feasibility(method_frame, path):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for row, method in enumerate(METHODS):
        frame = method_frame[method_frame["method"] == method]
        for column, (metric, title) in enumerate(
            (("pitch_feasible_fraction", "Pitch feasibility"), ("joint_feasible_fraction", "Joint-speed feasibility"))
        ):
            axis = axes[row, column]
            for index, train_sigma in enumerate(sorted(frame["train_sigma"].unique())):
                group = frame[frame["train_sigma"] == train_sigma].sort_values("eval_sigma")
                mean = 100.0 * group[f"{metric}_across_seed_mean"].to_numpy()
                standard_deviation = 100.0 * group[f"{metric}_across_seed_sd"].to_numpy()
                x = group["eval_sigma"].to_numpy()
                color = sigma_color(train_sigma, index)
                axis.plot(x, mean, marker="o", color=color, linewidth=2.0, label=sigma_label(train_sigma))
                axis.fill_between(x, mean - standard_deviation, mean + standard_deviation, color=color, alpha=0.18)
            axis.set_title(f"{method}: {title}", fontweight="bold")
            axis.set_xlabel("Evaluation gravity σ (m/s²)")
            axis.set_ylabel("Feasible episodes (%)")
            axis.set_ylim(-3, 103)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
    figure.suptitle("Individual Constraint Robustness: Mean Across Seeds ± 1 SD", fontweight="bold")
    save_figure(figure, path)


def plot_raw_state_metrics(method_frame, path):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metrics = (
        ("pitch_abs_max_deg_mean", "Mean episode-maximum |pitch| (degrees)"),
        ("joint_max_utilization_mean", "Mean maximum joint-speed / threshold"),
    )
    for row, method in enumerate(METHODS):
        frame = method_frame[method_frame["method"] == method]
        for column, (metric, ylabel) in enumerate(metrics):
            axis = axes[row, column]
            for index, train_sigma in enumerate(sorted(frame["train_sigma"].unique())):
                group = frame[frame["train_sigma"] == train_sigma].sort_values("eval_sigma")
                x = group["eval_sigma"].to_numpy()
                mean = group[f"{metric}_across_seed_mean"].to_numpy()
                standard_deviation = group[f"{metric}_across_seed_sd"].to_numpy()
                color = sigma_color(train_sigma, index)
                axis.plot(x, mean, marker="o", color=color, linewidth=2.0, label=sigma_label(train_sigma))
                axis.fill_between(x, mean - standard_deviation, mean + standard_deviation, color=color, alpha=0.18)
            if metric == "pitch_abs_max_deg_mean":
                axis.axhline(99.0, color="black", linestyle="--", linewidth=1.2)
            else:
                axis.axhline(1.1, color="black", linestyle="--", linewidth=1.2)
            axis.set_title(method, fontweight="bold")
            axis.set_xlabel("Evaluation gravity σ (m/s²)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
    figure.suptitle("Raw Physical-State Robustness: Mean Across Seeds ± 1 SD", fontweight="bold")
    save_figure(figure, path)


def plot_retention(retention_frame, path):
    if retention_frame.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for row, method in enumerate(METHODS):
        frame = retention_frame[retention_frame["method"] == method]
        for index, train_sigma in enumerate(sorted(frame["train_sigma"].unique())):
            group = frame[frame["train_sigma"] == train_sigma]
            color = sigma_color(train_sigma, index)
            label = sigma_label(train_sigma)
            for column, metric, scale in (
                (0, "reward_retention_fraction", 100.0),
                (1, "state_feasibility_change", 100.0),
            ):
                axis = axes[row, column]
                means = []
                deviations = []
                x_values = []
                for eval_sigma, values in group.groupby("eval_sigma", sort=True):
                    x_values.append(eval_sigma)
                    means.append(scale * values[metric].mean())
                    deviations.append(scale * values[metric].std(ddof=1))
                x_values = np.asarray(x_values)
                means = np.asarray(means)
                deviations = np.nan_to_num(np.asarray(deviations))
                axis.plot(x_values, means, marker="o", linewidth=2.0, color=color, label=label)
                axis.fill_between(x_values, means - deviations, means + deviations, color=color, alpha=0.10)
        axes[row, 0].axhline(100.0, color="black", linestyle="--", linewidth=1.2)
        axes[row, 0].set_ylabel("Reward retained from σ=0 (%)")
        axes[row, 1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        axes[row, 1].set_ylabel("State-feasibility change from σ=0 (points)")
        for column in range(2):
            axes[row, column].set_title(method, fontweight="bold")
            axes[row, column].set_xlabel("Evaluation gravity σ (m/s²)")
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend(frameon=False)
    figure.suptitle("Robustness Retention Relative to Nominal Evaluation: Mean ± 1 SD", fontweight="bold")
    save_figure(figure, path)


def plot_gravity_validation(episode_frame, path):
    first = episode_frame[
        (episode_frame["method"] == episode_frame.iloc[0]["method"])
        & (episode_frame["train_sigma"] == episode_frame.iloc[0]["train_sigma"])
        & (episode_frame["training_seed"] == episode_frame.iloc[0]["training_seed"])
    ]
    grouped = first.groupby("eval_sigma")["gravity_noise_sd"]
    x = np.asarray(sorted(grouped.groups), dtype=np.float64)
    mean = np.asarray([grouped.get_group(value).mean() for value in x])
    standard_deviation = np.asarray([grouped.get_group(value).std(ddof=1) for value in x])
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.errorbar(x, mean, yerr=standard_deviation, marker="o", capsize=4, linewidth=2.0, label="Observed per-episode SD")
    axis.plot(x, x, color="black", linestyle="--", label="Requested σ")
    axis.set_xlabel("Requested evaluation gravity σ (m/s²)")
    axis.set_ylabel("Observed gravity-noise SD (m/s²)")
    axis.set_title("Gravity Perturbation Validation", fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save_figure(figure, path)


def markdown_summary(method_frame, policy_frame, paired_frame, args):
    lines = [
        "# HalfCheetah Gravity-Robustness Evaluation",
        "",
        f"Each selected policy was evaluated deterministically for {args.episodes} episodes per gravity level on reset seeds {args.base_seed}–{args.base_seed + args.episodes - 1}.",
        "Gravity was independently redrawn before every MuJoCo step and therefore affected the next state, reward, and costs.",
        "",
        "## Method means across training seeds",
        "",
        "| Method | Train σ | Eval σ | Reward | State feasible | Mean max cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in method_frame.sort_values(["method", "train_sigma", "eval_sigma"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['train_sigma']:.2f} | {row['eval_sigma']:.2f} | "
            f"{row['reward_mean_across_seed_mean']:.1f} | "
            f"{100.0 * row['state_feasible_fraction_across_seed_mean']:.1f}% | "
            f"{row['combined_cost_max_mean_across_seed_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Robustness is shown as degradation as evaluation σ increases, not performance at a single perturbed setting.",
            "- The nominal policies are required baselines for deciding whether perturbed training helped.",
            "- The 100 reset episodes characterize each fixed policy. Algorithm-level uncertainty comes from independent training seeds.",
            f"- There are {policy_frame['training_seed'].nunique()} training seeds per condition, so seed-level conclusions remain limited.",
            "- Exact sign-flip p-values and paired seed differences are in `seed_level_paired_training_comparisons.csv`.",
            "- `rho` is kept separate from gravity σ: σ controls environment variation; rho controls the existing IPM critic-norm term.",
            "",
            "## Primary outputs",
            "",
            "- `episode_metrics.csv`: one row per policy, evaluation σ, and reset episode.",
            "- `policy_summary.csv`: 100-episode summary for every selected policy.",
            "- `method_summary_across_training_seeds.csv`: method means, SDs, and 95% t intervals across training seeds.",
            "- `robustness_retention.csv`: reward retention and feasibility change relative to σ=0.",
            "- `seed_level_paired_method_comparisons.csv`: scalar-versus-multi comparisons across matched training seeds.",
            "- `within_seed_matched_reset_tests.csv`: fixed-policy paired-reset diagnostics, kept separate from seed-level inference.",
            "- `selected_episode_timestep_metrics.csv`: raw gravity, pitch, joint speed, action, reward, and cost traces.",
        ]
    )
    if not paired_frame.empty and not (paired_frame["exact_sign_flip_pvalue"] < 0.05).any():
        lines.extend(["", "No paired training-seed comparison reached p < 0.05 under the exact sign-flip test."])
    return "\n".join(lines) + "\n"


def main():
    args = parse_arguments()
    validate_arguments(args)
    repo_root = Path(__file__).resolve().parent
    specs = build_policy_specs(repo_root, args)
    if args.dry_run:
        for spec in specs:
            status = "FOUND" if spec.actor_path.exists() else "MISSING"
            print(f"[{status}] {spec.condition}, seed {spec.training_seed}: {spec.actor_path}")
        return
    validate_policy_specs(specs)
    timestamp = datetime.now().strftime("evaluation_%Y%m%d_%H%M%S")
    output_directory = repo_root / args.output_root / timestamp
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=False)
    episode_rows = []
    selected_rows = []
    total = len(specs) * len(args.eval_sigmas)
    completed = 0
    for spec in specs:
        for eval_sigma in args.eval_sigmas:
            completed += 1
            print(
                f"[{completed}/{total}] {spec.condition}, seed {spec.training_seed}, "
                f"eval σ={eval_sigma:g}"
            )
            episode_rows.extend(evaluate_policy(spec, eval_sigma, args, selected_rows))
    episode_frame = pd.DataFrame(episode_rows)
    selected_frame = pd.DataFrame(selected_rows)
    policy_frame = summarize_policies(episode_frame)
    method_frame = summarize_methods(policy_frame)
    paired_frame = paired_training_tests(policy_frame)
    method_paired_frame = paired_method_tests(policy_frame)
    matched_reset_frame = matched_reset_tests(episode_frame)
    retention_frame = retention_table(policy_frame)
    episode_frame.to_csv(output_directory / "episode_metrics.csv", index=False)
    selected_frame.to_csv(output_directory / "selected_episode_timestep_metrics.csv", index=False)
    policy_frame.to_csv(output_directory / "policy_summary.csv", index=False)
    method_frame.to_csv(output_directory / "method_summary_across_training_seeds.csv", index=False)
    paired_frame.to_csv(output_directory / "seed_level_paired_training_comparisons.csv", index=False)
    method_paired_frame.to_csv(output_directory / "seed_level_paired_method_comparisons.csv", index=False)
    matched_reset_frame.to_csv(output_directory / "within_seed_matched_reset_tests.csv", index=False)
    retention_frame.to_csv(output_directory / "robustness_retention.csv", index=False)
    plot_robustness_metric(
        method_frame,
        "reward_mean",
        "Mean 100-episode reward",
        "Reward Robustness: Mean Across Training Seeds ± 1 SD",
        figure_directory / "robustness_reward_mean_sd.png",
    )
    plot_robustness_metric(
        method_frame,
        "state_feasible_fraction",
        "State-feasible episodes (%)",
        "Pitch + Joint Feasibility Robustness: Mean ± 1 SD",
        figure_directory / "robustness_state_feasibility_mean_sd.png",
        percent=True,
    )
    plot_robustness_metric(
        method_frame,
        "combined_cost_max_mean",
        "Mean episode-maximum combined cost",
        "Combined-Cost Robustness: Mean Across Training Seeds ± 1 SD",
        figure_directory / "robustness_combined_cost_mean_sd.png",
        threshold=args.persistent_eps,
    )
    plot_constraint_feasibility(
        method_frame,
        figure_directory / "robustness_individual_constraints_mean_sd.png",
    )
    plot_raw_state_metrics(
        method_frame,
        figure_directory / "robustness_raw_physical_metrics_mean_sd.png",
    )
    plot_retention(retention_frame, figure_directory / "robustness_retention_mean_sd.png")
    plot_gravity_validation(episode_frame, figure_directory / "gravity_perturbation_validation.png")
    metadata = {
        "episodes_per_policy_per_eval_sigma": args.episodes,
        "evaluation_seed_start": args.base_seed,
        "evaluation_seed_end": args.base_seed + args.episodes - 1,
        "train_sigmas": args.train_sigmas,
        "eval_sigmas": args.eval_sigmas,
        "rho": args.rho,
        "training_seeds": args.seeds,
        "robust_runs": args.robust_runs,
        "nominal_included": not args.exclude_nominal,
        "gravity_distribution": "Normal(base_gravity_z, eval_sigma)",
        "gravity_resampling": "every_step_before_simulation",
        "cost_threshold": args.persistent_eps,
        "actor_evaluation": "deterministic_mean_action",
    }
    (output_directory / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_directory / "README_results.md").write_text(
        markdown_summary(method_frame, policy_frame, paired_frame, args),
        encoding="utf-8",
    )
    manifest = pd.DataFrame(
        [
            ("robustness_reward_mean_sd.png", "Reward versus evaluation gravity sigma; line is across-seed mean and band is ±1 SD."),
            ("robustness_state_feasibility_mean_sd.png", "Pitch-plus-joint feasibility versus evaluation gravity sigma."),
            ("robustness_combined_cost_mean_sd.png", "Mean episode-maximum combined cost versus evaluation gravity sigma."),
            ("robustness_individual_constraints_mean_sd.png", "Pitch and joint feasibility shown separately."),
            ("robustness_raw_physical_metrics_mean_sd.png", "Raw maximum torso pitch and joint-speed threshold utilization."),
            ("robustness_retention_mean_sd.png", "Reward retention and feasibility loss relative to sigma zero."),
            ("gravity_perturbation_validation.png", "Observed per-episode gravity-noise standard deviation versus requested sigma."),
        ],
        columns=("figure", "description"),
    )
    manifest.to_csv(figure_directory / "figure_manifest.csv", index=False)
    print(f"Saved robustness evaluation to {output_directory}")


if __name__ == "__main__":
    main()
