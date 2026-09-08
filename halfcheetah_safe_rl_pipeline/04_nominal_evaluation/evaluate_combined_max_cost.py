import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT = Path("/home/bannanakid/RCRL_on_RCMDP")
sys.path.insert(0, str(PROJECT))

from envs.half_cheetah_combined_max_cost import HalfCheetahCombinedMaxCost
from configs.halfcheetah_combined_safety import (
    PITCH_THRESHOLD,
    PITCH_SCALE,
    JOINT_NAMES,
    JOINT_SPEED_THRESHOLDS,
    JOINT_SPEED_SCALES,
    PERSISTENT_EPS,
    COMBINED_COST_VERSION,
)


ENV_NAME = "HalfCheetahCombinedMaxCost"
BEST_NAME = "Best_CombinedMaxCost"


class ActorGaussian(nn.Module):
    def __init__(self, state_dim, hidden_width, action_dim, max_action):
        super().__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.activate_func = nn.ReLU()

    def forward(self, state):
        state = self.activate_func(self.fc1(state))
        state = self.activate_func(self.fc2(state))
        state = self.activate_func(self.fc3(state))
        return self.max_action * torch.tanh(self.mean_layer(state))


def unwrap_reset(reset_result):
    result = reset_result
    while isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float64)


def load_actor(path, env):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "actor_state_dict" in state:
        state = state["actor_state_dict"]
    actor = ActorGaussian(
        int(state["fc1.weight"].shape[1]),
        int(state["fc1.weight"].shape[0]),
        int(state["mean_layer.weight"].shape[0]),
        float(env.action_space.high[0]),
    )
    actor.load_state_dict(state)
    actor.eval()
    return actor


def longest_true_run(values):
    best = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def violation_events(values):
    values = np.asarray(values, dtype=bool)
    if len(values) == 0:
        return 0
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))


def collect_policy(actor, label, first_seed, episodes):
    env = HalfCheetahCombinedMaxCost()
    episode_rows = []
    step_episode = []
    step_seed = []
    step_index = []
    step_reward = []
    step_combined_cost = []
    step_pitch_cost = []
    step_joint_cost = []
    step_pitch_abs = []
    step_joint_speeds = []
    step_max_joint_speed = []

    for ep in range(episodes):
        seed = first_seed + ep
        obs = unwrap_reset(env.reset(seed=seed))
        start_x = float(env.sim.data.qpos[0])
        previous_x = start_x
        traveled = 0.0
        rewards = []
        combined_costs = []
        pitch_costs = []
        joint_costs = []
        pitches = []
        joint_speeds = []
        t = 0
        done = False

        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(x).cpu().numpy().reshape(-1)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs_next, reward, combined_cost, truncated, terminated, info = env.step(action)
            done = bool(truncated or terminated)

            current_x = float(info.get("xpos", env.sim.data.qpos[0]))
            traveled += abs(current_x - previous_x)
            previous_x = current_x

            pitch_abs = float(info["torso_pitch_abs"])
            pitch_cost = float(info["pitch_cost"])
            joint_cost = float(info["joint_speed_cost"])
            speeds = np.asarray(info["joint_speeds"], dtype=np.float64).copy()

            rewards.append(float(reward))
            combined_costs.append(float(combined_cost))
            pitch_costs.append(pitch_cost)
            joint_costs.append(joint_cost)
            pitches.append(pitch_abs)
            joint_speeds.append(speeds)

            step_episode.append(ep + 1)
            step_seed.append(seed)
            step_index.append(t)
            step_reward.append(float(reward))
            step_combined_cost.append(float(combined_cost))
            step_pitch_cost.append(pitch_cost)
            step_joint_cost.append(joint_cost)
            step_pitch_abs.append(pitch_abs)
            step_joint_speeds.append(speeds)
            step_max_joint_speed.append(float(np.max(speeds)))

            obs = obs_next
            t += 1

        rewards = np.asarray(rewards, dtype=np.float64)
        combined_costs = np.asarray(combined_costs, dtype=np.float64)
        pitch_costs = np.asarray(pitch_costs, dtype=np.float64)
        joint_costs = np.asarray(joint_costs, dtype=np.float64)
        pitches = np.asarray(pitches, dtype=np.float64)
        joint_speeds = np.asarray(joint_speeds, dtype=np.float64)

        combined_flags = combined_costs > 0.0
        combined_severe = combined_costs > float(PERSISTENT_EPS)
        pitch_flags = pitches > float(PITCH_THRESHOLD)
        joint_flags = np.any(joint_speeds > np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64), axis=1)
        end_x = float(env.sim.data.qpos[0])

        episode_rows.append(
            {
                "policy": label,
                "episode": ep + 1,
                "seed": seed,
                "reward": float(np.sum(rewards)),
                "total_combined_cost": float(np.sum(combined_costs)),
                "total_pitch_cost": float(np.sum(pitch_costs)),
                "total_joint_speed_cost": float(np.sum(joint_costs)),
                "max_combined_cost": float(np.max(combined_costs)),
                "max_pitch_cost": float(np.max(pitch_costs)),
                "max_joint_speed_cost": float(np.max(joint_costs)),
                "combined_step_violation_rate": float(np.mean(combined_flags)),
                "combined_severe_step_rate": float(np.mean(combined_severe)),
                "pitch_step_violation_rate": float(np.mean(pitch_flags)),
                "joint_step_violation_rate": float(np.mean(joint_flags)),
                "combined_any_violation": bool(np.any(combined_flags)),
                "combined_any_severe_violation": bool(np.any(combined_severe)),
                "pitch_any_violation": bool(np.any(pitch_flags)),
                "joint_any_violation": bool(np.any(joint_flags)),
                "combined_longest_violation_run": longest_true_run(combined_flags),
                "pitch_longest_violation_run": longest_true_run(pitch_flags),
                "joint_longest_violation_run": longest_true_run(joint_flags),
                "combined_violation_events": violation_events(combined_flags),
                "pitch_violation_events": violation_events(pitch_flags),
                "joint_violation_events": violation_events(joint_flags),
                "pitch_p95": float(np.percentile(pitches, 95)),
                "pitch_p99": float(np.percentile(pitches, 99)),
                "maximum_absolute_pitch": float(np.max(pitches)),
                "maximum_joint_speed": float(np.max(joint_speeds)),
                "net_displacement": float(end_x - start_x),
                "absolute_displacement": float(abs(end_x - start_x)),
                "traveled_distance": float(traveled),
                "episode_length": int(t),
            }
        )

        print(
            f"{label}: episode {ep + 1}/{episodes} seed={seed} "
            f"reward={np.sum(rewards):.3f} combined_cost={np.sum(combined_costs):.6f} "
            f"pitch_cost={np.sum(pitch_costs):.6f} joint_cost={np.sum(joint_costs):.6f} "
            f"max_combined={np.max(combined_costs):.6f}"
        )

    env.close()

    return {
        "episode_table": pd.DataFrame(episode_rows),
        "episode": np.asarray(step_episode, dtype=np.int64),
        "seed": np.asarray(step_seed, dtype=np.int64),
        "step": np.asarray(step_index, dtype=np.int64),
        "reward": np.asarray(step_reward, dtype=np.float64),
        "combined_cost": np.asarray(step_combined_cost, dtype=np.float64),
        "pitch_cost": np.asarray(step_pitch_cost, dtype=np.float64),
        "joint_cost": np.asarray(step_joint_cost, dtype=np.float64),
        "pitch_abs": np.asarray(step_pitch_abs, dtype=np.float64),
        "joint_speeds": np.asarray(step_joint_speeds, dtype=np.float64),
        "maximum_joint_speed": np.asarray(step_max_joint_speed, dtype=np.float64),
    }


def summarize(data):
    episodes = data["episode_table"]
    combined_flags = data["combined_cost"] > 0.0
    severe_flags = data["combined_cost"] > float(PERSISTENT_EPS)
    pitch_flags = data["pitch_abs"] > float(PITCH_THRESHOLD)
    joint_flags = np.any(
        data["joint_speeds"] > np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64),
        axis=1,
    )
    return {
        "mean_episode_reward": float(episodes["reward"].mean()),
        "std_episode_reward": float(episodes["reward"].std(ddof=0)),
        "median_episode_reward": float(episodes["reward"].median()),
        "mean_episode_total_combined_cost": float(episodes["total_combined_cost"].mean()),
        "mean_episode_total_pitch_cost": float(episodes["total_pitch_cost"].mean()),
        "mean_episode_total_joint_speed_cost": float(episodes["total_joint_speed_cost"].mean()),
        "maximum_one_step_combined_cost": float(np.max(data["combined_cost"])),
        "maximum_one_step_pitch_cost": float(np.max(data["pitch_cost"])),
        "maximum_one_step_joint_speed_cost": float(np.max(data["joint_cost"])),
        "combined_step_violation_rate": float(np.mean(combined_flags)),
        "combined_severe_step_rate": float(np.mean(severe_flags)),
        "pitch_step_violation_rate": float(np.mean(pitch_flags)),
        "joint_step_violation_rate": float(np.mean(joint_flags)),
        "combined_episode_violation_frequency": float(episodes["combined_any_violation"].mean()),
        "combined_severe_episode_frequency": float(episodes["combined_any_severe_violation"].mean()),
        "pitch_episode_violation_frequency": float(episodes["pitch_any_violation"].mean()),
        "joint_episode_violation_frequency": float(episodes["joint_any_violation"].mean()),
        "combined_longest_violation_run": int(episodes["combined_longest_violation_run"].max()),
        "pitch_longest_violation_run": int(episodes["pitch_longest_violation_run"].max()),
        "joint_longest_violation_run": int(episodes["joint_longest_violation_run"].max()),
        "mean_combined_violation_events_per_episode": float(episodes["combined_violation_events"].mean()),
        "mean_pitch_violation_events_per_episode": float(episodes["pitch_violation_events"].mean()),
        "mean_joint_violation_events_per_episode": float(episodes["joint_violation_events"].mean()),
        "mean_absolute_pitch": float(np.mean(data["pitch_abs"])),
        "pitch_p95": float(np.percentile(data["pitch_abs"], 95)),
        "pitch_p99": float(np.percentile(data["pitch_abs"], 99)),
        "maximum_absolute_pitch": float(np.max(data["pitch_abs"])),
        "mean_maximum_joint_speed": float(np.mean(data["maximum_joint_speed"])),
        "maximum_joint_speed": float(np.max(data["maximum_joint_speed"])),
        "mean_net_displacement": float(episodes["net_displacement"].mean()),
        "mean_absolute_displacement": float(episodes["absolute_displacement"].mean()),
        "mean_traveled_distance": float(episodes["traveled_distance"].mean()),
    }


def per_joint_statistics(data, label):
    speeds = data["joint_speeds"]
    thresholds = np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64)
    episode_ids = np.unique(data["episode"])
    rows = []
    for i, name in enumerate(JOINT_NAMES):
        flags = speeds[:, i] > thresholds[i]
        episode_flags = []
        for ep in episode_ids:
            mask = data["episode"] == ep
            episode_flags.append(bool(np.any(flags[mask])))
        rows.append(
            {
                "policy": label,
                "joint": name,
                "threshold": float(thresholds[i]),
                "scale": float(np.asarray(JOINT_SPEED_SCALES, dtype=np.float64)[i]),
                "mean_speed": float(np.mean(speeds[:, i])),
                "p95": float(np.percentile(speeds[:, i], 95)),
                "p99": float(np.percentile(speeds[:, i], 99)),
                "maximum": float(np.max(speeds[:, i])),
                "step_violation_rate": float(np.mean(flags)),
                "episode_violation_frequency": float(np.mean(episode_flags)),
            }
        )
    return pd.DataFrame(rows)


def reduction(before, after):
    if abs(before) <= 1e-12:
        return np.nan
    return 100.0 * (before - after) / abs(before)


def make_plots(output, summaries, episode_tables, per_joint):
    labels = list(summaries.keys())
    reward_values = [summaries[label]["mean_episode_reward"] for label in labels]
    cost_values = [summaries[label]["mean_episode_total_combined_cost"] for label in labels]
    violation_values = [summaries[label]["combined_step_violation_rate"] for label in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, reward_values)
    plt.ylabel("Mean held-out episode reward")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output / "mean_reward_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, cost_values)
    plt.ylabel("Mean held-out episode combined max cost")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output / "mean_combined_cost_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, violation_values)
    plt.ylabel("Combined step violation rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output / "combined_step_violation_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    for label, table in episode_tables.items():
        plt.plot(table["seed"], table["reward"], label=label)
    plt.xlabel("Held-out reset seed")
    plt.ylabel("Episode reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "episode_reward_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    for label, table in episode_tables.items():
        plt.plot(table["seed"], table["total_combined_cost"], label=label)
    plt.xlabel("Held-out reset seed")
    plt.ylabel("Episode total combined max cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "episode_combined_cost_comparison.png", dpi=160)
    plt.close()

    x = np.arange(len(JOINT_NAMES), dtype=np.float64)
    width = 0.18
    plt.figure(figsize=(12, 6))
    for j, label in enumerate(labels):
        table = per_joint[per_joint["policy"] == label].set_index("joint").loc[JOINT_NAMES]
        plt.bar(x + (j - 1.5) * width, table["p99"].to_numpy(), width, label=label)
    plt.scatter(x, np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64), marker="_", s=350, label="Threshold")
    plt.xticks(x, JOINT_NAMES, rotation=25)
    plt.ylabel("P99 absolute joint speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "per_joint_p99_comparison.png", dpi=160)
    plt.close()


def make_training_plots(output, run):
    data_dir = PROJECT / "data_train" / ENV_NAME / COMBINED_COST_VERSION / f"run{run}"
    reward_path = data_dir / "evaluate_rewards.npy"
    cost_path = data_dir / "evaluate_costs.npy"
    max_path = data_dir / "evaluate_max_costs.npy"
    if not (reward_path.exists() and cost_path.exists() and max_path.exists()):
        return
    rewards = np.load(reward_path)
    costs = np.load(cost_path)
    max_costs = np.load(max_path)
    episodes = np.arange(len(rewards)) * 100

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards)
    plt.xlabel("Training episode")
    plt.ylabel("Evaluation reward")
    plt.tight_layout()
    plt.savefig(output / "training_evaluation_reward.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, costs)
    plt.xlabel("Training episode")
    plt.ylabel("Evaluation combined cost")
    plt.tight_layout()
    plt.savefig(output / "training_evaluation_cost.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, max_costs)
    plt.axhline(float(PERSISTENT_EPS), linestyle="--")
    plt.xlabel("Training episode")
    plt.ylabel("Evaluation maximum one-step combined cost")
    plt.tight_layout()
    plt.savefig(output / "training_evaluation_max_cost.png", dpi=160)
    plt.close()


def main(args):
    if args.first_seed != 1005 or args.episodes != 50:
        raise ValueError("Official held-out evaluation must use exactly seeds 1005 through 1054")

    probe_env = HalfCheetahCombinedMaxCost()

    paths = {
        "Unconstrained": PROJECT / "models" / "HalfCheetahUnconstrained" / f"run{args.run}" / "Best_Unconstrained_actor",
        "Pitch only": PROJECT / "models" / "HalfCheetahPitchCost" / "wrapped_inversion_90deg_v3" / f"run{args.run}" / "Best_PitchCost_actor",
        "Joint-speed only": PROJECT / "models" / "HalfCheetahJointSpeedCost" / f"run{args.run}" / "Best_JointSpeedCost_actor",
        "Combined max": PROJECT / "models" / ENV_NAME / COMBINED_COST_VERSION / f"run{args.run}" / f"{BEST_NAME}_actor",
    }

    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")

    actors = {label: load_actor(path, probe_env) for label, path in paths.items()}
    probe_env.close()

    output = PROJECT / "plot_data" / ENV_NAME / COMBINED_COST_VERSION / f"run{args.run}" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)

    print("Cost version:", COMBINED_COST_VERSION)
    print("Aggregation: max")
    print("Pitch threshold:", PITCH_THRESHOLD)
    print("Pitch scale:", PITCH_SCALE)
    print("Joint thresholds:", np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64))
    print("Joint scales:", np.asarray(JOINT_SPEED_SCALES, dtype=np.float64))
    print("Held-out seeds:", args.first_seed, "through", args.first_seed + args.episodes - 1)

    data = {}
    for label, actor in actors.items():
        data[label] = collect_policy(actor, label, args.first_seed, args.episodes)

    summaries = {label: summarize(values) for label, values in data.items()}
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv")

    episode_results = pd.concat([values["episode_table"] for values in data.values()], ignore_index=True)
    episode_results.to_csv(output / "episode_results.csv", index=False)

    key_metrics = [
        "reward",
        "total_combined_cost",
        "total_pitch_cost",
        "total_joint_speed_cost",
        "max_combined_cost",
        "combined_step_violation_rate",
        "combined_severe_step_rate",
        "pitch_step_violation_rate",
        "joint_step_violation_rate",
        "absolute_displacement",
        "traveled_distance",
    ]
    comparison_parts = []
    for metric in key_metrics:
        pivot = episode_results.pivot(index="seed", columns="policy", values=metric)
        pivot.columns = [f"{metric}_{column.lower().replace(' ', '_').replace('-', '_')}" for column in pivot.columns]
        comparison_parts.append(pivot)
    episode_comparison = pd.concat(comparison_parts, axis=1).reset_index()
    episode_comparison.to_csv(output / "episode_comparison.csv", index=False)

    per_joint = pd.concat(
        [per_joint_statistics(values, label) for label, values in data.items()],
        ignore_index=True,
    )
    per_joint.to_csv(output / "per_joint_statistics.csv", index=False)

    arrays = {
        "pitch_threshold": np.asarray(PITCH_THRESHOLD, dtype=np.float64),
        "pitch_scale": np.asarray(PITCH_SCALE, dtype=np.float64),
        "joint_speed_thresholds": np.asarray(JOINT_SPEED_THRESHOLDS, dtype=np.float64),
        "joint_speed_scales": np.asarray(JOINT_SPEED_SCALES, dtype=np.float64),
        "persistent_eps": np.asarray(PERSISTENT_EPS, dtype=np.float64),
        "cost_version": np.asarray(COMBINED_COST_VERSION),
        "first_seed": np.asarray(args.first_seed, dtype=np.int64),
        "episodes": np.asarray(args.episodes, dtype=np.int64),
    }
    for label, values in data.items():
        prefix = label.lower().replace(" ", "_").replace("-", "_")
        arrays[f"{prefix}_seed"] = values["seed"]
        arrays[f"{prefix}_step"] = values["step"]
        arrays[f"{prefix}_reward"] = values["reward"]
        arrays[f"{prefix}_combined_cost"] = values["combined_cost"]
        arrays[f"{prefix}_pitch_cost"] = values["pitch_cost"]
        arrays[f"{prefix}_joint_cost"] = values["joint_cost"]
        arrays[f"{prefix}_pitch_abs"] = values["pitch_abs"]
        arrays[f"{prefix}_joint_speeds"] = values["joint_speeds"]
    np.savez_compressed(output / "evaluation_arrays.npz", **arrays)

    make_plots(
        output,
        summaries,
        {label: values["episode_table"] for label, values in data.items()},
        per_joint,
    )
    make_training_plots(output, args.run)

    baseline = summaries["Unconstrained"]
    combined = summaries["Combined max"]
    reward_retention = 100.0 * combined["mean_episode_reward"] / baseline["mean_episode_reward"]
    combined_cost_reduction = reduction(
        baseline["mean_episode_total_combined_cost"],
        combined["mean_episode_total_combined_cost"],
    )
    combined_step_reduction = reduction(
        baseline["combined_step_violation_rate"],
        combined["combined_step_violation_rate"],
    )
    pitch_step_reduction = reduction(
        baseline["pitch_step_violation_rate"],
        combined["pitch_step_violation_rate"],
    )
    joint_step_reduction = reduction(
        baseline["joint_step_violation_rate"],
        combined["joint_step_violation_rate"],
    )
    traveled_retention = 100.0 * combined["mean_traveled_distance"] / baseline["mean_traveled_distance"]

    print()
    print("=" * 80)
    print("OFFICIAL COMBINED MAX HELD-OUT RESULT")
    print("=" * 80)
    print(summary.to_string())
    print()
    print(f"Combined reward retention: {reward_retention:.2f}%")
    print(f"Combined mean-cost reduction: {combined_cost_reduction:.2f}%")
    print(f"Combined step-violation reduction: {combined_step_reduction:.2f}%")
    print(f"Pitch step-violation reduction: {pitch_step_reduction:.2f}%")
    print(f"Joint-speed step-violation reduction: {joint_step_reduction:.2f}%")
    print(f"Traveled-distance retention: {traveled_retention:.2f}%")
    print("Output directory:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--first_seed", type=int, default=1005)
    parser.add_argument("--episodes", type=int, default=50)
    main(parser.parse_args())
