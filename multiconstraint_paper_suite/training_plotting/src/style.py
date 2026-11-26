from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {
    "Ours": "#1f77b4",
    "Multi Constraint": "#1f77b4",
    "Multi-Constraint": "#1f77b4",
    "RP-CRL": "#1f77b4",
    "Combined Cost": "#ff7f0e",
    "Surrogate Obj": "#ff7f0e",
    "PD": "#2ca02c",
    "FAC": "#d62728",
    "RCRL": "#9467bd",
    "RESPO": "#8c564b",
    "Baseline": "black",
}

FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def color(label, idx=0):
    if label in {"Multi Constraint (NP)", "Multi-Constraint (NP)", "Multi (NP)"}:
        return "#ff7f0e"
    if label in {"Multi Constraint (P+R)", "Multi-Constraint (P+R)", "Multi (P+R)"}:
        return "#1f77b4"
    for key, value in COLORS.items():
        if label == key or label.startswith(key + " ") or label.startswith(key + "("):
            return value
    return FALLBACK[idx % len(FALLBACK)]


def smooth(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size < window:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode="valid")


def training_rc():
    plt.rcParams.update({"font.size": 100, "lines.linewidth": 15, "font.weight": "bold"})


def inference_rc():
    plt.rcParams.update({"font.size": 100, "lines.linewidth": 15, "font.weight": "bold"})


def spines(ax):
    for s in ax.spines.values():
        s.set_linewidth(15)


def legends(labels, prefix):
    labels = list(dict.fromkeys(labels))
    handles = [Line2D([0], [0], color=color(label, i), linewidth=15, label=label) for i, label in enumerate(labels)]
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
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


def safety(ax, threshold, fixed=None, linewidth=15):
    if fixed is None:
        y0, y1 = ax.get_ylim()
    else:
        y0, y1 = fixed
    if y1 <= threshold:
        y1 = threshold + max(abs(threshold) * 0.2, 0.1)
    if y0 >= threshold:
        y0 = min(0.0, threshold - max(abs(threshold) * 0.2, 0.1))
    ax.axhline(y=threshold, color="black", linestyle="--", linewidth=linewidth)
    ax.axhspan(threshold, y1, color="red", alpha=0.1)
    ax.axhspan(y0, threshold, color="blue", alpha=0.1)
