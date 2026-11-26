from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np


def load_run(run: Path):
    reward = np.load(run / "evaluate_rewards.npy").astype(float).reshape(-1)
    max_cost = np.load(run / "evaluate_max_costs.npy").astype(float)
    if max_cost.ndim == 1:
        worst = max_cost.reshape(-1)
        components = None
    else:
        components = max_cost
        worst = np.max(max_cost, axis=1)
    n = min(len(reward), len(worst))
    return reward[:n], worst[:n], None if components is None else components[:n]


def select_checkpoint(reward_matrix, cost_matrix, threshold):
    mean_reward = reward_matrix.mean(axis=0)
    mean_cost = cost_matrix.mean(axis=0)
    all_seed_feasible = np.all(cost_matrix <= threshold, axis=0)
    feasible_idx = np.flatnonzero(all_seed_feasible)
    if feasible_idx.size:
        idx = int(feasible_idx[np.argmax(mean_reward[feasible_idx])])
        reason = "highest mean reward among checkpoints feasible in both seeds"
    else:
        min_cost = float(np.min(mean_cost))
        candidates = np.flatnonzero(np.isclose(mean_cost, min_cost))
        idx = int(candidates[np.argmax(mean_reward[candidates])])
        reason = "no jointly feasible checkpoint; lowest mean worst cost, reward tie-break"
    return idx, reason


def main():
    ap = argparse.ArgumentParser(description="Analyze a method from exactly two seed run folders.")
    ap.add_argument("--run", action="append", required=True, help="Run directory containing evaluate_*.npy; pass twice")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--label", default="method")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if len(args.run) != 2:
        raise SystemExit("Pass exactly two --run arguments")

    loaded = [load_run(Path(x)) for x in args.run]
    n = min(len(x[0]) for x in loaded)
    rewards = np.stack([x[0][:n] for x in loaded], axis=0)
    costs = np.stack([x[1][:n] for x in loaded], axis=0)
    idx, reason = select_checkpoint(rewards, costs, args.threshold)

    payload = {
        "label": args.label,
        "runs": args.run,
        "threshold": args.threshold,
        "num_aligned_checkpoints": n,
        "selected_checkpoint_index_zero_based": idx,
        "selected_checkpoint_number_one_based": idx + 1,
        "selection_reason": reason,
        "selected_mean_reward": float(rewards[:, idx].mean()),
        "selected_reward_sd": float(rewards[:, idx].std(ddof=0)),
        "selected_mean_worst_max_cost": float(costs[:, idx].mean()),
        "selected_worst_cost_sd": float(costs[:, idx].std(ddof=0)),
        "selected_seed_rewards": rewards[:, idx].tolist(),
        "selected_seed_worst_max_costs": costs[:, idx].tolist(),
        "jointly_feasible": bool(np.all(costs[:, idx] <= args.threshold)),
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
