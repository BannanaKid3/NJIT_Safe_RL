from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np


def load(run):
    run = Path(run)
    r = np.load(run / "evaluate_rewards.npy").astype(float).reshape(-1)
    c = np.load(run / "evaluate_max_costs.npy").astype(float)
    if c.ndim > 1:
        c = np.max(c, axis=1)
    return r, c.reshape(-1)


def aggregate(run_paths):
    arr = [load(x) for x in run_paths]
    n = min(len(r) for r, _ in arr)
    rewards = np.stack([r[:n] for r, _ in arr])
    costs = np.stack([c[:n] for _, c in arr])
    return rewards, costs


def select(rewards, costs, threshold):
    mr = rewards.mean(0)
    mc = costs.mean(0)
    feasible = np.all(costs <= threshold, axis=0)
    inds = np.flatnonzero(feasible)
    if inds.size:
        idx = int(inds[np.argmax(mr[inds])])
    else:
        inds = np.flatnonzero(np.isclose(mc, mc.min()))
        idx = int(inds[np.argmax(mr[inds])])
    return idx, float(mr[idx]), float(mc[idx]), bool(feasible[idx])


def main():
    ap = argparse.ArgumentParser(description="Two-seed Multi vs Combined summary. This reports metrics; it does not decide the paper conclusion for you.")
    ap.add_argument("--multi-run", action="append", required=True)
    ap.add_argument("--combined-run", action="append", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if len(a.multi_run) != 2 or len(a.combined_run) != 2:
        raise SystemExit("Provide exactly two --multi-run and two --combined-run folders")
    mr, mc = aggregate(a.multi_run)
    cr, cc = aggregate(a.combined_run)
    mi = select(mr, mc, a.threshold)
    ci = select(cr, cc, a.threshold)
    dominated = ci[1] <= mi[1] and ci[2] >= mi[2]
    out = {
        "threshold": a.threshold,
        "multi": {"checkpoint_index": mi[0], "mean_reward": mi[1], "mean_worst_max_cost": mi[2], "jointly_feasible": mi[3]},
        "combined": {"checkpoint_index": ci[0], "mean_reward": ci[1], "mean_worst_max_cost": ci[2], "jointly_feasible": ci[3]},
        "combined_pareto_dominated_at_selected_checkpoints": bool(dominated),
        "definition": "combined reward <= multi reward AND combined worst max cost >= multi worst max cost"
    }
    text = json.dumps(out, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
