from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

METHOD_COLORS = {
    "Surrogate Obj": "C0",
    "Ours": "C1",
    "Combined": "C2",
    "Multi": "C1",
}


def smooth_preserve_prefix(values, window=10, prefix=10):
    x = np.asarray(values, dtype=float)
    y = x.copy()
    if window <= 1:
        return y
    for i in range(max(prefix, 0), len(x)):
        lo = max(0, i - window + 1)
        y[i] = np.mean(x[lo:i + 1])
    return y


def load_metric(run, name):
    return np.load(Path(run) / name).astype(float)


def align(arrays):
    n = min(a.shape[0] for a in arrays)
    return [a[:n] for a in arrays]


def aggregate_two(arrays, smooth=True):
    if len(arrays) != 2:
        raise ValueError("Paper aggregation in this project expects exactly two seeds")
    arrays = align(arrays)
    stack = np.stack(arrays, axis=0)
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=0)
    if smooth:
        if mean.ndim == 1:
            mean = smooth_preserve_prefix(mean)
            sd = smooth_preserve_prefix(sd)
        else:
            mean = np.stack([smooth_preserve_prefix(mean[:, j]) for j in range(mean.shape[1])], axis=1)
            sd = np.stack([smooth_preserve_prefix(sd[:, j]) for j in range(sd.shape[1])], axis=1)
    return mean, sd


def shade_safety(ax, threshold):
    ymin, ymax = ax.get_ylim()
    lo = min(ymin, 0.0)
    hi = max(ymax, threshold * 1.2 if threshold > 0 else ymax)
    ax.axhspan(lo, threshold, color="lightblue", alpha=0.18)
    ax.axhspan(threshold, hi, color="mistyrose", alpha=0.22)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=2.5, label="Safety threshold")


def plot_scalar(methods, ylabel, title, out, threshold=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    for label, arrays in methods.items():
        mean, sd = aggregate_two(arrays)
        x = np.arange(1, len(mean) + 1)
        color = METHOD_COLORS.get(label)
        ax.plot(x, mean, linewidth=3.0, label=label, color=color)
        ax.fill_between(x, mean - 1.5 * sd, mean + 1.5 * sd, alpha=0.18, color=color)
    ax.set_xlabel("Evaluation checkpoint", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    if threshold is not None:
        shade_safety(ax, threshold)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Two-seed paper plotter: first 10 unsmoothed, then trailing-window 10; band is ±1.5 SD.")
    ap.add_argument("--method", action="append", nargs=3, metavar=("LABEL", "RUN1", "RUN2"), required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--outdir", default="paper_plots")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    reward_methods = {}
    worst_methods = {}
    component_methods = {}
    for label, r1, r2 in args.method:
        reward_methods[label] = [load_metric(r1, "evaluate_rewards.npy").reshape(-1), load_metric(r2, "evaluate_rewards.npy").reshape(-1)]
        costs = [load_metric(r1, "evaluate_max_costs.npy"), load_metric(r2, "evaluate_max_costs.npy")]
        if costs[0].ndim == 1:
            worst_methods[label] = [c.reshape(-1) for c in costs]
        else:
            worst_methods[label] = [np.max(c, axis=1) for c in costs]
            component_methods[label] = costs

    plot_scalar(reward_methods, "Reward", "Evaluation Reward", out / "reward.png")
    plot_scalar(worst_methods, "Worst Max Cost", "Worst Constraint Cost", out / "max_cost.png", threshold=args.threshold)

    # If a method exposes vector costs, produce individual component plots too.
    max_dim = max((arrs[0].shape[1] for arrs in component_methods.values()), default=0)
    for j in range(max_dim):
        methods = {}
        for label, arrs in component_methods.items():
            if arrs[0].shape[1] > j and arrs[1].shape[1] > j:
                methods[label] = [arrs[0][:, j], arrs[1][:, j]]
        if methods:
            plot_scalar(methods, f"Max C{j+1}", f"Constraint C{j+1}", out / f"cost_component_C{j+1}.png", threshold=args.threshold)

    print("WROTE", out.resolve())


if __name__ == "__main__":
    main()
