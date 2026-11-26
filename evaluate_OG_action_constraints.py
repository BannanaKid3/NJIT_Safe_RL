import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from configs.halfcheetah_combined_safety import (
    JOINT_NAMES,
    JOINT_SPEED_SCALES,
    JOINT_SPEED_THRESHOLDS,
    PERSISTENT_EPS,
    PITCH_SCALE,
    PITCH_THRESHOLD,
)
from envs.half_cheetah import ACTION_TORQUE_THRESHOLD, HalfCheetahWithPos


class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        return (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)


class ActorGaussian(nn.Module):
    def __init__(self, state_dim, hidden_width, action_dim, max_action, activation):
        super().__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.activate_func = nn.Tanh() if activation == "tanh" else nn.ReLU()

    def forward(self, state):
        state = self.activate_func(self.fc1(state))
        state = self.activate_func(self.fc2(state))
        state = self.activate_func(self.fc3(state))
        return self.max_action * torch.tanh(self.mean_layer(state))


def reset_observation(env, seed):
    np.random.seed(seed)
    result = env.reset(seed=seed)
    while isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float32)


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


def resolve_model(repo_root, run, model_prefix):
    if model_prefix:
        prefix = Path(model_prefix).expanduser()
        if not prefix.is_absolute():
            prefix = repo_root / prefix
    else:
        prefix = repo_root / "models" / "HalfCheetahWithPos" / f"run{run}" / "Best_RCAC"
    if prefix.name.endswith("_actor"):
        actor_path = prefix
        prefix = Path(str(prefix)[:-6])
    else:
        actor_path = Path(str(prefix) + "_actor")
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
    return prefix.resolve(), actor_path.resolve()


def load_actor(actor_path, max_action, activation, device):
    state = clean_state_dict(torch_load(actor_path, device))
    required = {"fc1.weight", "fc2.weight", "fc3.weight", "mean_layer.weight"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"The checkpoint is not the expected Gaussian actor: {missing}")
    state_dim = int(state["fc1.weight"].shape[1])
    hidden_width = int(state["fc1.weight"].shape[0])
    action_dim = int(state["mean_layer.weight"].shape[0])
    actor = ActorGaussian(state_dim, hidden_width, action_dim, max_action, activation).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor, state_dim, hidden_width, action_dim


def load_state_normalizer(model_prefix):
    path = Path(str(model_prefix) + "_state_norm")
    if not path.exists():
        return None, None
    with path.open("rb") as handle:
        normalizer = pickle.load(handle)
    return normalizer, path


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def diagnostic_values(env):
    pitch_raw = float(env.sim.data.qpos[2])
    pitch = wrap_angle(pitch_raw)
    pitch_abs = abs(pitch)
    pitch_excess = max(pitch_abs - float(PITCH_THRESHOLD), 0.0)
    pitch_cost = pitch_excess / float(PITCH_SCALE)
    joint_velocities = np.asarray(env.sim.data.qvel[3:9], dtype=np.float64).copy()
    joint_speeds = np.abs(joint_velocities)
    thresholds = np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64)
    scales = np.asarray(JOINT_SPEED_SCALES, dtype=np.float64)
    joint_excess = np.maximum(joint_speeds - thresholds, 0.0)
    joint_costs = joint_excess / scales
    joint_speed_cost = float(np.max(joint_costs))
    combined_cost = float(max(pitch_cost, joint_speed_cost))
    return {
        "torso_pitch_raw": pitch_raw,
        "torso_pitch": pitch,
        "torso_pitch_abs": pitch_abs,
        "pitch_cost": float(pitch_cost),
        "joint_velocities": joint_velocities,
        "joint_speeds": joint_speeds,
        "joint_costs": joint_costs,
        "joint_speed_cost": joint_speed_cost,
        "combined_cost": combined_cost,
    }


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    margin = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, mean - margin, mean + margin


def evaluate(args, repo_root, model_prefix, actor, state_normalizer, device):
    env = HalfCheetahWithPos()
    env.action_space.seed(args.base_seed)
    records = []
    summaries = []
    joint_names = list(JOINT_NAMES)
    for episode_index in range(1, args.episodes + 1):
        evaluation_seed = args.base_seed + episode_index - 1
        observation = reset_observation(env, evaluation_seed)
        episode_records = []
        done = False
        timestep = 0
        while not done and timestep < args.max_episode_steps:
            actor_observation = observation
            if state_normalizer is not None:
                actor_observation = state_normalizer(actor_observation, update=False)
            state_tensor = torch.as_tensor(actor_observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().reshape(-1)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_observation, reward, action_cost, truncated, terminated, info = env.step(action)
            done = bool(truncated or terminated)
            diagnostic = diagnostic_values(env)
            row = {
                "episode": episode_index,
                "evaluation_seed": evaluation_seed,
                "timestep": timestep + 1,
                "time_seconds": float((timestep + 1) * env.dt),
                "reward": float(reward),
                "action_constraint_cost": float(action_cost),
                "maximum_absolute_action": float(np.max(np.abs(action))),
                "torso_pitch_raw": diagnostic["torso_pitch_raw"],
                "torso_pitch": diagnostic["torso_pitch"],
                "torso_pitch_abs": diagnostic["torso_pitch_abs"],
                "pitch_cost": diagnostic["pitch_cost"],
                "joint_speed_cost": diagnostic["joint_speed_cost"],
                "combined_diagnostic_cost": diagnostic["combined_cost"],
            }
            for index, name in enumerate(joint_names):
                row[f"action_{index + 1}"] = float(action[index])
                row[f"{name}_velocity"] = float(diagnostic["joint_velocities"][index])
                row[f"{name}_speed"] = float(diagnostic["joint_speeds"][index])
                row[f"{name}_cost"] = float(diagnostic["joint_costs"][index])
            episode_records.append(row)
            records.append(row)
            observation = np.asarray(next_observation, dtype=np.float32)
            timestep += 1
        episode_frame = pd.DataFrame(episode_records)
        summary = {
            "episode": episode_index,
            "evaluation_seed": evaluation_seed,
            "episode_steps": len(episode_frame),
            "episode_return": float(episode_frame["reward"].sum()),
            "action_constraint_total_cost": float(episode_frame["action_constraint_cost"].sum()),
            "action_constraint_max_cost": float(episode_frame["action_constraint_cost"].max()),
            "maximum_absolute_action": float(episode_frame["maximum_absolute_action"].max()),
            "pitch_total_cost": float(episode_frame["pitch_cost"].sum()),
            "pitch_max_cost": float(episode_frame["pitch_cost"].max()),
            "joint_speed_total_cost": float(episode_frame["joint_speed_cost"].sum()),
            "joint_speed_max_cost": float(episode_frame["joint_speed_cost"].max()),
            "combined_total_cost": float(episode_frame["combined_diagnostic_cost"].sum()),
            "combined_max_cost": float(episode_frame["combined_diagnostic_cost"].max()),
            "action_constraint_feasible": bool(episode_frame["action_constraint_cost"].max() <= args.persistent_eps),
            "diagnostic_pitch_feasible": bool(episode_frame["pitch_cost"].max() <= args.persistent_eps),
            "diagnostic_joint_speed_feasible": bool(episode_frame["joint_speed_cost"].max() <= args.persistent_eps),
            "diagnostic_combined_feasible": bool(episode_frame["combined_diagnostic_cost"].max() <= args.persistent_eps),
        }
        for name in joint_names:
            summary[f"{name}_maximum_speed"] = float(episode_frame[f"{name}_speed"].max())
            summary[f"{name}_maximum_cost"] = float(episode_frame[f"{name}_cost"].max())
        summaries.append(summary)
        print(
            f"[{episode_index}/{args.episodes}] seed={evaluation_seed} "
            f"reward={summary['episode_return']:.3f} "
            f"action_max={summary['action_constraint_max_cost']:.6f} "
            f"pitch_max={summary['pitch_max_cost']:.6f} "
            f"joint_max={summary['joint_speed_max_cost']:.6f} "
            f"combined_safe={summary['diagnostic_combined_feasible']}"
        )
    env.close()
    return pd.DataFrame(records), pd.DataFrame(summaries)


def plot_selected_episode(timestep_frame, output_directory, episode_index):
    selected = timestep_frame[timestep_frame["episode"] == episode_index].sort_values("timestep")
    if selected.empty:
        raise ValueError(f"Plot episode {episode_index} is not present")
    time = selected["time_seconds"].to_numpy()
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(time, selected["torso_pitch"], color="#2F6BFF", linewidth=1.7, label="Wrapped torso pitch")
    axis.axhline(float(PITCH_THRESHOLD), color="black", linestyle="--", linewidth=1.3, label="Pitch thresholds")
    axis.axhline(-float(PITCH_THRESHOLD), color="black", linestyle="--", linewidth=1.3)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Torso pitch (rad)")
    axis.set_title(f"Action-Constrained Policy Final Inference: Torso Pitch, Episode {episode_index}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "torso_pitch.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(JOINT_NAMES)))
    figure, axes = plt.subplots(6, 1, figsize=(12, 17), sharex=True)
    for index, (name, threshold, color) in enumerate(zip(JOINT_NAMES, JOINT_SPEED_THRESHOLDS, colors)):
        values = selected[f"{name}_speed"].to_numpy()
        axes[index].plot(time, values, color=color, linewidth=1.5, label=name.replace("_", " "))
        axes[index].axhline(float(threshold), color="black", linestyle="--", linewidth=1.2, label=f"P99 threshold = {threshold:.2f}")
        axes[index].set_ylabel("Speed (rad/s)")
        axes[index].grid(alpha=0.25)
        axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"Six Joint Speeds During Final Inference, Episode {episode_index}", y=0.995)
    figure.tight_layout()
    figure.savefig(output_directory / "six_joint_speeds.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    for index, (name, threshold, color) in enumerate(zip(JOINT_NAMES, JOINT_SPEED_THRESHOLDS, colors), start=1):
        figure, axis = plt.subplots(figsize=(12, 5.2))
        axis.plot(time, selected[f"{name}_speed"], color=color, linewidth=1.7, label=name.replace("_", " "))
        axis.axhline(float(threshold), color="black", linestyle="--", linewidth=1.3, label=f"P99 threshold = {threshold:.2f}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Absolute joint speed (rad/s)")
        axis.set_title(f"{name.replace('_', ' ').title()} Speed, Episode {episode_index}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_directory / f"joint_speed_{index}_{name}.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(time, selected["pitch_cost"], color="#8E5EA2", linewidth=1.7, label="Pitch cost")
    axes[1].plot(time, selected["joint_speed_cost"], color="#2CA02C", linewidth=1.7, label="Maximum joint-speed cost")
    axes[2].plot(time, selected["combined_diagnostic_cost"], color="#D62728", linewidth=1.7, label="Combined diagnostic cost")
    for axis in axes:
        axis.axhline(float(PERSISTENT_EPS), color="black", linestyle="--", linewidth=1.2, label="Diagnostic safety threshold")
        axis.set_ylabel("Normalized cost")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"Post-Hoc Pitch and Joint-Speed Costs, Episode {episode_index}", y=0.995)
    figure.tight_layout()
    figure.savefig(output_directory / "diagnostic_costs.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(6, 1, figsize=(12, 17), sharex=True)
    for index, (name, color) in enumerate(zip(JOINT_NAMES, colors)):
        axes[index].plot(time, selected[f"{name}_cost"], color=color, linewidth=1.5, label=name.replace("_", " "))
        axes[index].axhline(float(PERSISTENT_EPS), color="black", linestyle="--", linewidth=1.2, label="Diagnostic safety threshold")
        axes[index].set_ylabel("Cost")
        axes[index].grid(alpha=0.25)
        axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(f"Six Individual Joint-Speed Costs, Episode {episode_index}", y=0.995)
    figure.tight_layout()
    figure.savefig(output_directory / "six_individual_joint_costs.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time, selected["maximum_absolute_action"], color="#2F6BFF", linewidth=1.7, label="Maximum absolute action")
    axes[0].axhline(float(ACTION_TORQUE_THRESHOLD), color="black", linestyle="--", linewidth=1.2, label="Action-cost activation threshold")
    axes[0].axhline(float(ACTION_TORQUE_THRESHOLD + PERSISTENT_EPS), color="#B22222", linestyle=":", linewidth=1.4, label="Checkpoint feasibility boundary")
    axes[0].set_ylabel("Maximum |action|")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(time, selected["action_constraint_cost"], color="#F28E2B", linewidth=1.7, label="Training action cost")
    axes[1].axhline(float(PERSISTENT_EPS), color="black", linestyle="--", linewidth=1.2, label="Training safety threshold")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Action cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle(f"Original Action Constraint During Final Inference, Episode {episode_index}", y=0.995)
    figure.tight_layout()
    figure.savefig(output_directory / "action_constraint.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_episode_summary(summary_frame, output_directory):
    episodes = summary_frame["episode"].to_numpy()
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(episodes, summary_frame["episode_return"], color="#2F6BFF", marker="o", linewidth=1.5)
    axes[0].set_ylabel("Episode return")
    axes[0].grid(alpha=0.25)
    axes[1].plot(episodes, summary_frame["action_constraint_max_cost"], color="#F28E2B", marker="o", linewidth=1.5, label="Action constraint")
    axes[1].axhline(float(PERSISTENT_EPS), color="black", linestyle="--", linewidth=1.2, label="Safety threshold")
    axes[1].set_ylabel("Maximum action cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].plot(episodes, summary_frame["pitch_max_cost"], color="#8E5EA2", marker="o", linewidth=1.5, label="Pitch")
    axes[2].plot(episodes, summary_frame["joint_speed_max_cost"], color="#2CA02C", marker="s", linewidth=1.5, label="Joint speed")
    axes[2].plot(episodes, summary_frame["combined_max_cost"], color="#D62728", marker="D", linewidth=1.5, label="Combined")
    axes[2].axhline(float(PERSISTENT_EPS), color="black", linestyle="--", linewidth=1.2, label="Diagnostic safety threshold")
    axes[2].set_xlabel("Evaluation episode")
    axes[2].set_ylabel("Maximum diagnostic cost")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.suptitle("Action-Constrained Policy Evaluation Summary", y=0.995)
    figure.tight_layout()
    figure.savefig(output_directory / "evaluation_episode_summary.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--model_prefix", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=2301)
    parser.add_argument("--plot_episode", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--persistent_eps", type=float, default=float(PERSISTENT_EPS))
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if args.plot_episode < 1 or args.plot_episode > args.episodes:
        raise ValueError("plot_episode must be between 1 and episodes")
    repo_root = Path(__file__).resolve().parent
    model_prefix, actor_path = resolve_model(repo_root, args.run, args.model_prefix)
    device = torch.device(args.device)
    env_probe = HalfCheetahWithPos()
    max_action = float(env_probe.action_space.high[0])
    expected_state_dim = int(env_probe.observation_space.shape[0])
    expected_action_dim = int(env_probe.action_space.shape[0])
    env_probe.close()
    actor, state_dim, hidden_width, action_dim = load_actor(actor_path, max_action, args.activation, device)
    if state_dim != expected_state_dim or action_dim != expected_action_dim:
        raise ValueError(
            f"Checkpoint dimensions do not match HalfCheetahWithPos: "
            f"checkpoint=({state_dim}, {action_dim}), environment=({expected_state_dim}, {expected_action_dim})"
        )
    state_normalizer, state_normalizer_path = load_state_normalizer(model_prefix)
    if args.output_root:
        output_root = Path(args.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = repo_root / output_root
    else:
        output_root = repo_root / "analysis" / "outputs" / "action_constrained_final_evaluation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = output_root / f"run{args.run}" / f"evaluation_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=False)
    timestep_frame, summary_frame = evaluate(args, repo_root, model_prefix, actor, state_normalizer, device)
    timestep_frame.to_csv(output_directory / "timestep_metrics.csv", index=False)
    summary_frame.to_csv(output_directory / "episode_summary.csv", index=False)
    plot_selected_episode(timestep_frame, output_directory, args.plot_episode)
    plot_episode_summary(summary_frame, output_directory)
    reward_mean, reward_low, reward_high = mean_ci95(summary_frame["episode_return"])
    metadata = {
        "environment": "HalfCheetahWithPos",
        "training_constraint": "action_torque_excess",
        "action_torque_threshold": float(ACTION_TORQUE_THRESHOLD),
        "training_persistent_epsilon": float(args.persistent_eps),
        "actor_checkpoint": str(actor_path),
        "state_normalizer": str(state_normalizer_path) if state_normalizer_path is not None else None,
        "actor_activation": args.activation,
        "state_dimension": state_dim,
        "hidden_width": hidden_width,
        "action_dimension": action_dim,
        "evaluation_episodes": args.episodes,
        "evaluation_seed_start": args.base_seed,
        "evaluation_seed_end": args.base_seed + args.episodes - 1,
        "deterministic_mean_action": True,
        "plot_episode": args.plot_episode,
        "pitch_and_joint_costs_used_for_training": False,
        "pitch_and_joint_costs_are_post_hoc_diagnostics": True,
        "pitch_threshold": float(PITCH_THRESHOLD),
        "pitch_scale": float(PITCH_SCALE),
        "joint_names": list(JOINT_NAMES),
        "joint_speed_thresholds": np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64).tolist(),
        "joint_speed_scales": np.asarray(JOINT_SPEED_SCALES, dtype=np.float64).tolist(),
        "diagnostic_persistent_epsilon": float(PERSISTENT_EPS),
        "reward_mean": reward_mean,
        "reward_ci95_low": reward_low,
        "reward_ci95_high": reward_high,
        "action_constraint_feasible_fraction": float(summary_frame["action_constraint_feasible"].mean()),
        "diagnostic_pitch_feasible_fraction": float(summary_frame["diagnostic_pitch_feasible"].mean()),
        "diagnostic_joint_speed_feasible_fraction": float(summary_frame["diagnostic_joint_speed_feasible"].mean()),
        "diagnostic_combined_feasible_fraction": float(summary_frame["diagnostic_combined_feasible"].mean()),
    }
    with (output_directory / "evaluation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Reward mean: {reward_mean:.3f} 95% CI=[{reward_low:.3f}, {reward_high:.3f}]")
    print(f"Action-constraint feasibility: {100.0 * metadata['action_constraint_feasible_fraction']:.1f}%")
    print(f"Post-hoc pitch feasibility: {100.0 * metadata['diagnostic_pitch_feasible_fraction']:.1f}%")
    print(f"Post-hoc joint-speed feasibility: {100.0 * metadata['diagnostic_joint_speed_feasible_fraction']:.1f}%")
    print(f"Post-hoc combined feasibility: {100.0 * metadata['diagnostic_combined_feasible_fraction']:.1f}%")
    print(f"Outputs: {output_directory}")


if __name__ == "__main__":
    main()
