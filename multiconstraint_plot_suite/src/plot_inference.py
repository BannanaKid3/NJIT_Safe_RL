from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from style import inference_rc, color, smooth, spines, legends, safety


def load_result(path):
    p = Path(path)
    if p.is_dir():
        r = np.load(p / "rewards.npy").astype(float).reshape(-1)
        c = np.load(p / "max_costs.npy").astype(float)
    else:
        z = np.load(p)
        r = np.asarray(z["rewards"], dtype=float).reshape(-1)
        c = np.asarray(z["max_costs"], dtype=float)
    if c.ndim > 1:
        c = np.max(c, axis=1)
    return r, c.reshape(-1)


def parse(items):
    out = []
    for item in items:
        if len(item) < 2:
            raise SystemExit("Each --method needs a label and at least one result")
        out.append((item[0], item[1:]))
    return out


def mean_series(paths, metric):
    xs = [load_result(p)[0 if metric == "reward" else 1] for p in paths]
    n = min(len(x) for x in xs)
    return np.stack([x[:n] for x in xs], axis=0).mean(axis=0)


def save_plot(methods, metric, threshold, window, out):
    inference_rc()
    fig = plt.figure(figsize=(44, 28) if metric == "reward" else (42, 28))
    ax = fig.add_subplot(111)
    for i, (label, paths) in enumerate(methods):
        y = smooth(mean_series(paths, metric), window)
        ax.plot(range(len(y)), y, color=color(label, i), label=label)
    ax.set_xlabel("Episode", fontweight="bold", fontsize=130)
    ax.set_ylabel("Cumulative Reward" if metric == "reward" else "Max Cost", fontweight="bold", fontsize=130)
    if metric == "cost":
        safety(ax, threshold, fixed=None, linewidth=15)
    ax.grid(False)
    spines(ax)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="inference")
    a = ap.parse_args()
    methods = parse(a.method)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / a.prefix
    save_plot(methods, "reward", a.threshold, a.window, str(prefix) + "_rewards.png")
    save_plot(methods, "cost", a.threshold, a.window, str(prefix) + "_max_costs.png")
    legends([m[0] for m in methods], prefix)
    meta = [{"label": label, "results": paths} for label, paths in methods]
    (out / f"{a.prefix}_results.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
