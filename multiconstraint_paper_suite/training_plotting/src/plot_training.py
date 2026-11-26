from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from style import training_rc, color, smooth, spines, legends, safety
from data import load_run, scalar_cost, stats


def parse_methods(items):
    out = []
    for item in items:
        if len(item) < 2:
            raise SystemExit("Each --method needs a label and at least one run folder")
        out.append((item[0], item[1:]))
    return out


def aggregate(paths, key, max_episodes):
    vals = []
    raw = []
    for p in paths:
        r, c = load_run(p, max_episodes)
        x = r if key == "reward" else scalar_cost(c)
        vals.append(x)
        raw.append(c)
    mean, sd, n = stats(vals)
    return mean, sd, n, raw


def save_plot(methods, key, threshold, window, max_episodes, out, fixed_shade=None):
    training_rc()
    fig = plt.figure(figsize=(36, 28))
    ax = fig.add_subplot(111)
    for i, (label, paths) in enumerate(methods):
        mean, sd, _, _ = aggregate(paths, key, max_episodes)
        m = smooth(mean, window)
        s = smooth(sd, window)
        x = range(len(m))
        col = color(label, i)
        ax.plot(x, m, color=col, label=label)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.2)
    ax.set_xlabel("Episode", fontweight="bold", fontsize=130)
    ax.set_ylabel("Reward" if key == "reward" else "Max Cost", fontweight="bold", fontsize=130)
    if key == "cost":
        safety(ax, threshold, fixed=fixed_shade, linewidth=15)
    ax.grid(False)
    spines(ax)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_components(methods, threshold, window, max_episodes, outdir, prefix, fixed_shade=None):
    dims = 0
    cache = {}
    for label, paths in methods:
        cs = [load_run(p, max_episodes)[1] for p in paths]
        if cs and cs[0].ndim == 2:
            dims = max(dims, min(c.shape[1] for c in cs))
            cache[label] = cs
    for j in range(dims):
        training_rc()
        fig = plt.figure(figsize=(36, 28))
        ax = fig.add_subplot(111)
        used = 0
        for i, (label, paths) in enumerate(methods):
            if label not in cache:
                continue
            mean, sd, _ = stats([c[:, j] for c in cache[label]])
            m = smooth(mean, window)
            s = smooth(sd, window)
            x = range(len(m))
            col = color(label, i)
            ax.plot(x, m, color=col, label=label)
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.2)
            used += 1
        if not used:
            plt.close(fig)
            continue
        ax.set_xlabel("Episode", fontweight="bold", fontsize=130)
        ax.set_ylabel(f"Max C{j+1}", fontweight="bold", fontsize=130)
        safety(ax, threshold, fixed=fixed_shade, linewidth=15)
        ax.grid(False)
        spines(ax)
        fig.savefig(Path(outdir) / f"{prefix}_C{j+1}_max_cost.png", bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--window", type=int, default=80)
    ap.add_argument("--max-episodes", type=int, default=10000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="training")
    ap.add_argument("--components", action="store_true")
    ap.add_argument("--shade-min", type=float)
    ap.add_argument("--shade-max", type=float)
    a = ap.parse_args()
    methods = parse_methods(a.method)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checked = []
    for label, paths in methods:
        for p in paths:
            r, c = load_run(p)
            if len(r) < a.max_episodes:
                raise SystemExit(f"{p}: only {len(r)} episodes, expected at least {a.max_episodes}")
            checked.append({"label": label, "path": p, "episodes": len(r), "max_cost_shape": list(c.shape)})
    methods = [(label, paths) for label, paths in methods]
    fixed = None
    if a.shade_min is not None and a.shade_max is not None:
        fixed = (a.shade_min, a.shade_max)
    prefix = outdir / a.prefix
    save_plot(methods, "reward", a.threshold, a.window, a.max_episodes, str(prefix) + "_rewards.png")
    save_plot(methods, "cost", a.threshold, a.window, a.max_episodes, str(prefix) + "_max_costs.png", fixed)
    legends([m[0] for m in methods], prefix)
    if a.components:
        save_components(methods, a.threshold, a.window, a.max_episodes, outdir, a.prefix, fixed)
    (outdir / f"{a.prefix}_runs.json").write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
