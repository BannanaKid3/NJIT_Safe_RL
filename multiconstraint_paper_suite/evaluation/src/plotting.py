from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "Ours": "#1f77b4",
    "Surrogate Obj": "#ff7f0e",
    "Multi Constraint": "#1f77b4",
    "Combined Cost": "#ff7f0e",
    "PD": "#2ca02c",
    "FAC": "#d62728",
    "RCRL": "#9467bd",
    "RESPO": "#8c564b",
}


def smooth(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < window:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode="valid")


def color(label, index):
    return COLORS.get(label, list(COLORS.values())[index % len(COLORS)])


def style():
    plt.rcParams.update({"font.size": 100, "lines.linewidth": 15, "font.weight": "bold"})


def set_spines(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(15)


def save_legends(labels, prefix):
    handles = [Line2D([0], [0], color=color(label, i), linewidth=15, label=label) for i, label in enumerate(labels)]
    fig = plt.figure(figsize=(36, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles=handles, labels=labels, loc="center", ncol=len(labels), frameon=False, fontsize=90)
    fig.savefig(str(prefix) + "_legend_horizontal.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles=handles, labels=labels, loc="center", ncol=1, frameon=False, fontsize=90)
    fig.savefig(str(prefix) + "_legend_vertical.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def method_series(entries, key):
    arrays = [np.asarray(entry["result"][key], dtype=float) for entry in entries]
    n = min(len(array) for array in arrays)
    arrays = np.stack([array[:n] for array in arrays], axis=0)
    return np.mean(arrays, axis=0), np.std(arrays, axis=0, ddof=0)


def main_plot(methods, key, threshold, window, path, ylabel):
    style()
    fig = plt.figure(figsize=(44, 28) if key == "rewards" else (42, 28))
    ax = fig.add_subplot(111)
    for i, (label, entries) in enumerate(methods.items()):
        if key == "component_max_costs":
            arrays = [np.max(np.asarray(entry["result"][key], dtype=float), axis=1) for entry in entries]
            n = min(len(array) for array in arrays)
            values = np.stack([array[:n] for array in arrays], axis=0).mean(axis=0)
        else:
            values, _ = method_series(entries, key)
        values = smooth(values, window)
        ax.plot(range(len(values)), values, color=color(label, i))
    if key != "rewards":
        y0, y1 = ax.get_ylim()
        if y1 <= threshold:
            y1 = threshold + max(0.1, abs(threshold) * 0.2)
        if y0 >= threshold:
            y0 = min(0.0, threshold - max(0.1, abs(threshold) * 0.2))
        ax.axhline(threshold, color="black", linestyle="--", linewidth=15)
        ax.axhspan(threshold, y1, color="red", alpha=0.1)
        ax.axhspan(y0, threshold, color="blue", alpha=0.1)
    ax.set_xlabel("Episode", fontweight="bold", fontsize=130)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=130)
    ax.grid(False)
    set_spines(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def component_plots(methods, threshold, window, outdir, names):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    first = next(iter(methods.values()))
    sample = np.asarray(first[0]["result"]["component_max_costs"])
    dim = sample.shape[1]
    names = names if names and len(names) == dim else [f"C{i + 1}" for i in range(dim)]
    for ci, name in enumerate(names):
        style()
        fig = plt.figure(figsize=(42, 28))
        ax = fig.add_subplot(111)
        for i, (label, entries) in enumerate(methods.items()):
            values, _ = method_series(entries, "component_max_costs")
            y = smooth(values[:, ci], window)
            ax.plot(range(len(y)), y, color=color(label, i))
        y0, y1 = ax.get_ylim()
        if y1 <= threshold:
            y1 = threshold + max(0.1, abs(threshold) * 0.2)
        if y0 >= threshold:
            y0 = min(0.0, threshold - max(0.1, abs(threshold) * 0.2))
        ax.axhline(threshold, color="black", linestyle="--", linewidth=15)
        ax.axhspan(threshold, y1, color="red", alpha=0.1)
        ax.axhspan(y0, threshold, color="blue", alpha=0.1)
        ax.set_xlabel("Episode", fontweight="bold", fontsize=130)
        ax.set_ylabel(f"Max {name}", fontweight="bold", fontsize=130)
        ax.grid(False)
        set_spines(ax)
        fig.savefig(outdir / f"max_{name.replace(' ', '_')}.png", bbox_inches="tight")
        plt.close(fig)


def total_component_plots(methods, window, outdir, names):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    first = next(iter(methods.values()))
    sample = np.asarray(first[0]["result"]["component_total_costs"])
    dim = sample.shape[1]
    names = names if names and len(names) == dim else [f"C{i + 1}" for i in range(dim)]
    for ci, name in enumerate(names):
        style()
        fig = plt.figure(figsize=(42, 28))
        ax = fig.add_subplot(111)
        for i, (label, entries) in enumerate(methods.items()):
            values, _ = method_series(entries, "component_total_costs")
            y = smooth(values[:, ci], window)
            ax.plot(range(len(y)), y, color=color(label, i))
        ax.set_xlabel("Episode", fontweight="bold", fontsize=130)
        ax.set_ylabel(f"Total {name}", fontweight="bold", fontsize=130)
        ax.grid(False)
        set_spines(ax)
        fig.savefig(outdir / f"total_{name.replace(' ', '_')}.png", bbox_inches="tight")
        plt.close(fig)


def save_bundle(methods, threshold, window, outdir, prefix="inference", component_names=None, totals=False, metadata=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix_path = outdir / prefix
    main_plot(methods, "rewards", threshold, window, str(prefix_path) + "_rewards.png", "Cumulative Reward")
    main_plot(methods, "component_max_costs", threshold, window, str(prefix_path) + "_max_costs.png", "Max Cost")
    save_legends(list(methods), prefix_path)
    component_plots(methods, threshold, window, outdir / "components", component_names)
    if totals:
        total_component_plots(methods, window, outdir / "components", component_names)
    meta = metadata or {}
    meta["methods"] = {label: [{"training_seed": x["seed"], "checkpoint": str(x["checkpoint"])} for x in entries] for label, entries in methods.items()}
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
