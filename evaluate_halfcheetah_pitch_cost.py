import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT = Path("/home/bannanakid/RCRL_on_RCMDP")
sys.path.insert(0, str(PROJECT))

from envs.half_cheetah_pitch_cost import HalfCheetahPitchCost, COST_VERSION
from configs.halfcheetah_safety_thresholds import PITCH_THRESHOLD, PITCH_SCALE


ENV_NAME = "HalfCheetahPitchCost"
BEST_NAME = "Best_PitchCost"


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
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def violation_events(values):
    values = np.asarray(values, dtype=bool)
    if len(values) == 0:
        return 0
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))


def collect_policy(actor, label, first_seed, episodes):
    env = HalfCheetahPitchCost()

    episode_rewards = []
    episode_costs = []
    episode_max_costs = []
    episode_lengths = []
    episode_net_displacements = []
    episode_absolute_displacements = []
    episode_traveled_distances = []
    episode_violation_flags = []
    episode_longest_violation_runs = []
    episode_violation_events = []

    all_pitch = []
    all_cost = []
    rows = []

    for ep in range(episodes):
        ep_seed = first_seed + ep
        obs = unwrap_reset(env.reset(seed=ep_seed))
        start_x = float(env.sim.data.qpos[0])
        previous_x = start_x
        total_reward = 0.0
        total_cost = 0.0
        max_cost = 0.0
        traveled = 0.0
        pitches = []
        costs = []
        t = 0
        done = False

        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(x).cpu().numpy().reshape(-1)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs_next, reward, cost, truncated, terminated, info = env.step(action)
            done = bool(truncated or terminated)

            current_x = float(info.get("xpos", env.sim.data.qpos[0]))
            pitch_abs = float(info["torso_pitch_abs"])

            traveled += abs(current_x - previous_x)
            previous_x = current_x
            total_reward += float(reward)
            total_cost += float(cost)
            max_cost = max(max_cost, float(cost))
            pitches.append(pitch_abs)
            costs.append(float(cost))
            obs = obs_next
            t += 1

        end_x = float(env.sim.data.qpos[0])
        pitches = np.asarray(pitches, dtype=np.float64)
        costs = np.asarray(costs, dtype=np.float64)
        flags = pitches > float(PITCH_THRESHOLD)

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_max_costs.append(max_cost)
        episode_lengths.append(t)
        episode_net_displacements.append(end_x - start_x)
        episode_absolute_displacements.append(abs(end_x - start_x))
        episode_traveled_distances.append(traveled)
        episode_violation_flags.append(bool(np.any(flags)))
        episode_longest_violation_runs.append(longest_true_run(flags))
        episode_violation_events.append(violation_events(flags))

        all_pitch.append(pitches)
        all_cost.append(costs)

        rows.append(
            {
                "policy": label,
                "episode": ep + 1,
                "seed": ep_seed,
                "reward": total_reward,
                "total_cost": total_cost,
                "max_one_step_cost": max_cost,
                "episode_length": t,
                "net_displacement": end_x - start_x,
                "absolute_displacement": abs(end_x - start_x),
                "traveled_distance": traveled,
                "step_violation_rate": float(np.mean(flags)),
                "any_violation": bool(np.any(flags)),
                "longest_violation_run": longest_true_run(flags),
                "violation_events": violation_events(flags),
                "pitch_p95": float(np.percentile(pitches, 95)),
                "pitch_p99": float(np.percentile(pitches, 99)),
                "maximum_absolute_pitch": float(np.max(pitches)),
            }
        )

        print(
            f"{label}: episode {ep + 1}/{episodes} seed={ep_seed} "
            f"reward={total_reward:.3f} total_cost={total_cost:.6f} "
            f"max_cost={max_cost:.6f} violated={bool(np.any(flags))}"
        )

    env.close()

    return {
        "episode_table": pd.DataFrame(rows),
        "episode_rewards": np.asarray(episode_rewards, dtype=np.float64),
        "episode_costs": np.asarray(episode_costs, dtype=np.float64),
        "episode_max_costs": np.asarray(episode_max_costs, dtype=np.float64),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
        "episode_net_displacements": np.asarray(episode_net_displacements, dtype=np.float64),
        "episode_absolute_displacements": np.asarray(episode_absolute_displacements, dtype=np.float64),
        "episode_traveled_distances": np.asarray(episode_traveled_distances, dtype=np.float64),
        "episode_violation_flags": np.asarray(episode_violation_flags, dtype=bool),
        "episode_longest_violation_runs": np.asarray(episode_longest_violation_runs, dtype=np.int64),
        "episode_violation_events": np.asarray(episode_violation_events, dtype=np.int64),
        "all_pitch": np.concatenate(all_pitch),
        "all_cost": np.concatenate(all_cost),
    }


def summarize(data):
    return {
        "mean_episode_reward": float(np.mean(data["episode_rewards"])),
        "std_episode_reward": float(np.std(data["episode_rewards"])),
        "median_episode_reward": float(np.median(data["episode_rewards"])),
        "mean_episode_total_cost": float(np.mean(data["episode_costs"])),
        "median_episode_total_cost": float(np.median(data["episode_costs"])),
        "maximum_one_step_cost": float(np.max(data["all_cost"])),
        "mean_absolute_pitch": float(np.mean(data["all_pitch"])),
        "pitch_p95": float(np.percentile(data["all_pitch"], 95)),
        "pitch_p99": float(np.percentile(data["all_pitch"], 99)),
        "maximum_absolute_pitch": float(np.max(data["all_pitch"])),
        "step_violation_rate": float(np.mean(data["all_pitch"] > float(PITCH_THRESHOLD))),
        "episode_violation_frequency": float(np.mean(data["episode_violation_flags"])),
        "longest_violation_run": int(np.max(data["episode_longest_violation_runs"])),
        "mean_longest_violation_run": float(np.mean(data["episode_longest_violation_runs"])),
        "mean_violation_events_per_episode": float(np.mean(data["episode_violation_events"])),
        "mean_net_displacement": float(np.mean(data["episode_net_displacements"])),
        "mean_absolute_displacement": float(np.mean(data["episode_absolute_displacements"])),
        "mean_traveled_distance": float(np.mean(data["episode_traveled_distances"])),
    }


def reduction(before, after):
    if abs(before) <= 1e-12:
        return np.nan
    return 100.0 * (before - after) / abs(before)


def main(args):
    if args.first_seed != 1005 or args.episodes != 50:
        raise ValueError("Official held-out evaluation must use exactly seeds 1005 through 1054")

    probe_env = HalfCheetahPitchCost()

    unconstrained_path = (
        PROJECT
        / "models"
        / "HalfCheetahUnconstrained"
        / f"run{args.run}"
        / "Best_Unconstrained_actor"
    )

    pitch_path = (
        PROJECT
        / "models"
        / ENV_NAME
        / COST_VERSION
        / f"run{args.run}"
        / f"{BEST_NAME}_actor"
    )

    output = (
        PROJECT
        / "plot_data"
        / ENV_NAME
        / COST_VERSION
        / f"run{args.run}"
        / "heldout_1005_1054"
    )
    output.mkdir(parents=True, exist_ok=True)

    if not unconstrained_path.exists():
        raise FileNotFoundError(unconstrained_path)
    if not pitch_path.exists():
        raise FileNotFoundError(
            f"{pitch_path}\nNo official feasible Best_PitchCost_actor was found."
        )

    unconstrained_actor = load_actor(unconstrained_path, probe_env)
    pitch_actor = load_actor(pitch_path, probe_env)
    probe_env.close()

    print("Cost version:", COST_VERSION)
    print("Pitch threshold:", PITCH_THRESHOLD)
    print("Pitch scale:", PITCH_SCALE)
    print("Held-out seeds:", args.first_seed, "through", args.first_seed + args.episodes - 1)
    print("Unconstrained actor:", unconstrained_path)
    print("Pitch actor:", pitch_path)

    unconstrained = collect_policy(
        unconstrained_actor,
        "Unconstrained",
        args.first_seed,
        args.episodes,
    )

    pitch_constrained = collect_policy(
        pitch_actor,
        "Pitch constrained",
        args.first_seed,
        args.episodes,
    )

    unconstrained_summary = summarize(unconstrained)
    pitch_summary = summarize(pitch_constrained)

    summary = pd.DataFrame(
        {
            "Unconstrained": unconstrained_summary,
            "Pitch constrained": pitch_summary,
        }
    )
    summary["Absolute change"] = summary["Pitch constrained"] - summary["Unconstrained"]
    summary["Percent change"] = np.where(
        np.abs(summary["Unconstrained"]) > 1e-12,
        100.0 * summary["Absolute change"] / np.abs(summary["Unconstrained"]),
        np.nan,
    )

    episode_table = unconstrained["episode_table"].drop(columns=["policy"]).copy()
    pitch_table = pitch_constrained["episode_table"].drop(columns=["policy"]).copy()
    episode_comparison = episode_table.merge(
        pitch_table,
        on=["episode", "seed"],
        suffixes=("_unconstrained", "_pitch"),
    )

    summary.to_csv(output / "summary.csv")
    episode_comparison.to_csv(output / "episode_comparison.csv", index=False)

    np.savez_compressed(
        output / "heldout_arrays.npz",
        seeds=np.arange(args.first_seed, args.first_seed + args.episodes, dtype=np.int64),
        unconstrained_rewards=unconstrained["episode_rewards"],
        pitch_rewards=pitch_constrained["episode_rewards"],
        unconstrained_costs=unconstrained["episode_costs"],
        pitch_costs=pitch_constrained["episode_costs"],
        unconstrained_max_costs=unconstrained["episode_max_costs"],
        pitch_max_costs=pitch_constrained["episode_max_costs"],
        unconstrained_net_displacements=unconstrained["episode_net_displacements"],
        pitch_net_displacements=pitch_constrained["episode_net_displacements"],
        pitch_threshold=np.asarray(PITCH_THRESHOLD, dtype=np.float64),
        pitch_scale=np.asarray(PITCH_SCALE, dtype=np.float64),
        cost_version=np.asarray(COST_VERSION),
    )

    reward_retention = 100.0 * (
        pitch_summary["mean_episode_reward"]
        / unconstrained_summary["mean_episode_reward"]
    )
    step_violation_reduction = reduction(
        unconstrained_summary["step_violation_rate"],
        pitch_summary["step_violation_rate"],
    )
    episode_violation_reduction = reduction(
        unconstrained_summary["episode_violation_frequency"],
        pitch_summary["episode_violation_frequency"],
    )
    total_cost_reduction = reduction(
        unconstrained_summary["mean_episode_total_cost"],
        pitch_summary["mean_episode_total_cost"],
    )

    print()
    print("=" * 72)
    print("OFFICIAL HELD-OUT RESULT")
    print("=" * 72)
    print(summary.to_string())
    print()
    print(f"Reward retention: {reward_retention:.2f}%")
    print(f"Step violation reduction: {step_violation_reduction:.2f}%")
    print(f"Episode violation-frequency reduction: {episode_violation_reduction:.2f}%")
    print(f"Mean episode pitch-cost reduction: {total_cost_reduction:.2f}%")
    print("Output directory:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--first_seed", type=int, default=1005)
    parser.add_argument("--episodes", type=int, default=50)
    main(parser.parse_args())