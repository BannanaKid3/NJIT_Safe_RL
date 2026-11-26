import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gravity_robustness_common import (
    MULTI_BASE_VERSION,
    SCALAR_BASE_VERSION,
    multi_gravity_version,
    parse_float_list,
    parse_int_list,
    scalar_gravity_version,
)


METHODS = ("Scalar combined", "Multi-constraint")
METHOD_DIRECTORIES = {
    "Scalar combined": "HalfCheetahCombinedMaxCost",
    "Multi-constraint": "HalfCheetahMultiConstraintCost",
}
COLORS = {0.0: "#6B7280", 0.5: "#2F6BFF", 0.7: "#F28E2B"}


@dataclass(frozen=True)
class TrainingCondition:
    method: str
    train_sigma: float
    version: str
    runs: tuple
    seeds: tuple


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sigmas", type=parse_float_list, default=[0.5, 0.7])
    parser.add_argument("--rho", type=float, default=0.001)
    parser.add_argument("--seeds", type=parse_int_list, default=[5, 11, 17, 23])
    parser.add_argument("--robust_runs", type=parse_int_list, default=[1, 2, 3, 4])
    parser.add_argument("--nominal_scalar_runs", type=parse_int_list, default=[4, 5, 6, 7])
    parser.add_argument("--nominal_multi_runs", type=parse_int_list, default=[4, 5, 6, 7])
    parser.add_argument("--include_nominal", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=100)
    parser.add_argument("--final_window", type=int, default=1000)
    parser.add_argument(
        "--output_root",
        default="analysis/outputs/gravity_robustness_training",
    )
    return parser.parse_args()


def validate_arguments(args):
    if len(args.seeds) != len(args.robust_runs):
        raise ValueError("seeds and robust_runs must have equal lengths")
    if args.include_nominal:
        if len(args.seeds) != len(args.nominal_scalar_runs):
            raise ValueError("seeds and nominal_scalar_runs must have equal lengths")
        if len(args.seeds) != len(args.nominal_multi_runs):
            raise ValueError("seeds and nominal_multi_runs must have equal lengths")
    if args.rho < 0.0:
        raise ValueError("rho cannot be negative")
    if any(value < 0.0 for value in args.train_sigmas):
        raise ValueError("train_sigmas cannot be negative")
    if args.smooth_window < 1 or args.final_window < 1:
        raise ValueError("window sizes must be positive")


def build_conditions(args):
    conditions = []
    if args.include_nominal:
        conditions.extend(
            [
                TrainingCondition(
                    "Scalar combined",
                    0.0,
                    SCALAR_BASE_VERSION,
                    tuple(args.nominal_scalar_runs),
                    tuple(args.seeds),
                ),
                TrainingCondition(
                    "Multi-constraint",
                    0.0,
                    MULTI_BASE_VERSION,
                    tuple(args.nominal_multi_runs),
                    tuple(args.seeds),
                ),
            ]
        )
    for sigma in args.train_sigmas:
        conditions.extend(
            [
                TrainingCondition(
                    "Scalar combined",
                    float(sigma),
                    scalar_gravity_version(sigma, args.rho),
                    tuple(args.robust_runs),
                    tuple(args.seeds),
                ),
                TrainingCondition(
                    "Multi-constraint",
                    float(sigma),
                    multi_gravity_version(sigma, args.rho),
                    tuple(args.robust_runs),
                    tuple(args.seeds),
                ),
            ]
        )
    return conditions


def condition_directory(repo_root, condition):
    return (
        repo_root
        / "plot_data"
        / METHOD_DIRECTORIES[condition.method]
        / condition.version
    )


def load_condition(repo_root, condition):
    rewards = []
    combined_max_costs = []
    pitch_max_costs = []
    joint_max_costs = []
    missing = []
    for run in condition.runs:
        run_directory = condition_directory(repo_root, condition) / f"run{run}"
        reward_path = run_directory / "episode_rewards.npy"
        cost_path = run_directory / "episode_max_costs.npy"
        if not reward_path.exists() or not cost_path.exists():
            missing.extend(path for path in (reward_path, cost_path) if not path.exists())
            continue
        reward = np.asarray(np.load(reward_path), dtype=np.float64).reshape(-1)
        raw_cost = np.asarray(np.load(cost_path), dtype=np.float64)
        if condition.method == "Multi-constraint":
            if raw_cost.ndim != 2 or raw_cost.shape[1] != 2:
                raise ValueError(f"Expected Nx2 multi costs in {cost_path}, got {raw_cost.shape}")
            pitch = raw_cost[:, 0]
            joint = raw_cost[:, 1]
            combined = np.max(raw_cost, axis=1)
            pitch_max_costs.append(pitch)
            joint_max_costs.append(joint)
        else:
            combined = raw_cost.reshape(-1)
        length = min(len(reward), len(combined))
        rewards.append(reward[:length])
        combined_max_costs.append(combined[:length])
    if missing:
        lines = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing training arrays:\n{lines}")
    return {
        "reward": rewards,
        "combined_max_cost": combined_max_costs,
        "pitch_max_cost": pitch_max_costs,
        "joint_max_cost": joint_max_costs,
    }


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    effective = min(int(window), len(values))
    if effective <= 1:
        return values.copy(), np.arange(1, len(values) + 1)
    kernel = np.ones(effective, dtype=np.float64) / effective
    smoothed = np.convolve(values, kernel, mode="valid")
    x = np.arange(effective, effective + len(smoothed))
    return smoothed, x


def aggregate_curves(curves, smooth_window):
    smoothed = []
    x_values = []
    for curve in curves:
        values, x = moving_average(curve, smooth_window)
        smoothed.append(values)
        x_values.append(x)
    minimum = min(len(values) for values in smoothed)
    matrix = np.vstack([values[:minimum] for values in smoothed])
    x = x_values[0][:minimum]
    mean = np.mean(matrix, axis=0)
    standard_deviation = np.std(matrix, axis=0, ddof=1) if len(matrix) > 1 else np.zeros(minimum)
    return x, mean, standard_deviation


def sigma_label(sigma):
    return "Nominal training" if sigma == 0.0 else f"Train σ={sigma:g}"


def color_for_sigma(sigma, index):
    return COLORS.get(float(sigma), plt.get_cmap("tab10")(index))


def plot_metric(
    data_by_condition,
    conditions,
    metric,
    ylabel,
    title,
    output_path,
    smooth_window,
    threshold=None,
):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True)
    for axis, method in zip(axes, METHODS):
        method_conditions = [item for item in conditions if item.method == method]
        for index, condition in enumerate(sorted(method_conditions, key=lambda item: item.train_sigma)):
            curves = data_by_condition[condition][metric]
            if not curves:
                continue
            x, mean, standard_deviation = aggregate_curves(curves, smooth_window)
            color = color_for_sigma(condition.train_sigma, index)
            axis.plot(x, mean, color=color, linewidth=2.2, label=sigma_label(condition.train_sigma))
            axis.fill_between(
                x,
                mean - standard_deviation,
                mean + standard_deviation,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        if threshold is not None:
            axis.axhline(threshold, color="black", linestyle="--", linewidth=1.4, label="Safety threshold")
        axis.set_title(method, fontweight="bold")
        axis.set_xlabel("Training episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(title, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_feasibility(data_by_condition, conditions, output_path, smooth_window, threshold):
    transformed = {}
    for condition in conditions:
        transformed[condition] = {
            "feasible": [
                100.0 * (np.asarray(curve) <= threshold).astype(np.float64)
                for curve in data_by_condition[condition]["combined_max_cost"]
            ]
        }
    plot_metric(
        transformed,
        conditions,
        "feasible",
        "Feasible training episodes (%)",
        "Training Feasibility: Mean Across Seeds ± 1 SD",
        output_path,
        smooth_window,
    )


def plot_multi_components(data_by_condition, conditions, output_path, smooth_window, threshold):
    multi_conditions = [item for item in conditions if item.method == "Multi-constraint"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True)
    for axis, metric, title in zip(
        axes,
        ("pitch_max_cost", "joint_max_cost"),
        ("Pitch maximum cost", "Joint-speed maximum cost"),
    ):
        for index, condition in enumerate(sorted(multi_conditions, key=lambda item: item.train_sigma)):
            curves = data_by_condition[condition][metric]
            if not curves:
                continue
            x, mean, standard_deviation = aggregate_curves(curves, smooth_window)
            color = color_for_sigma(condition.train_sigma, index)
            axis.plot(x, mean, color=color, linewidth=2.2, label=sigma_label(condition.train_sigma))
            axis.fill_between(x, mean - standard_deviation, mean + standard_deviation, color=color, alpha=0.18)
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1.4)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Episode maximum cost")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Multi-Constraint Training Components: Mean ± 1 SD", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def final_window_rows(data_by_condition, conditions, final_window, threshold):
    rows = []
    for condition in conditions:
        data = data_by_condition[condition]
        for seed, run, reward, cost in zip(
            condition.seeds,
            condition.runs,
            data["reward"],
            data["combined_max_cost"],
        ):
            length = min(final_window, len(reward), len(cost))
            rows.append(
                {
                    "method": condition.method,
                    "train_sigma": condition.train_sigma,
                    "training_seed": seed,
                    "run": run,
                    "final_window_episodes": length,
                    "reward_mean": float(np.mean(reward[-length:])),
                    "reward_sd": float(np.std(reward[-length:], ddof=1)) if length > 1 else 0.0,
                    "combined_max_cost_mean": float(np.mean(cost[-length:])),
                    "feasible_fraction": float(np.mean(cost[-length:] <= threshold)),
                }
            )
    return rows


def main():
    args = parse_arguments()
    validate_arguments(args)
    repo_root = Path(__file__).resolve().parent
    conditions = build_conditions(args)
    data_by_condition = {condition: load_condition(repo_root, condition) for condition in conditions}
    timestamp = datetime.now().strftime("evaluation_%Y%m%d_%H%M%S")
    output_directory = repo_root / args.output_root / timestamp
    output_directory.mkdir(parents=True, exist_ok=False)
    threshold = 0.1
    plot_metric(
        data_by_condition,
        conditions,
        "reward",
        "Episode reward",
        "Training Reward: Mean Across Seeds ± 1 SD",
        output_directory / "training_reward_mean_sd.png",
        args.smooth_window,
    )
    plot_metric(
        data_by_condition,
        conditions,
        "combined_max_cost",
        "Episode maximum combined cost",
        "Training Maximum Cost: Mean Across Seeds ± 1 SD",
        output_directory / "training_combined_max_cost_mean_sd.png",
        args.smooth_window,
        threshold=threshold,
    )
    plot_feasibility(
        data_by_condition,
        conditions,
        output_directory / "training_feasibility_mean_sd.png",
        args.smooth_window,
        threshold,
    )
    plot_multi_components(
        data_by_condition,
        conditions,
        output_directory / "training_multi_components_mean_sd.png",
        args.smooth_window,
        threshold,
    )
    rows = final_window_rows(data_by_condition, conditions, args.final_window, threshold)
    with (output_directory / "final_training_window_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = [
        {"figure": "training_reward_mean_sd.png", "description": "Smoothed mean reward across training seeds with ±1 SD shading."},
        {"figure": "training_combined_max_cost_mean_sd.png", "description": "Smoothed mean episode-maximum combined cost with ±1 SD shading."},
        {"figure": "training_feasibility_mean_sd.png", "description": "Smoothed percentage of feasible training episodes with ±1 SD shading."},
        {"figure": "training_multi_components_mean_sd.png", "description": "Separate multi-policy pitch and joint maximum costs with ±1 SD shading."},
    ]
    with (output_directory / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("figure", "description"))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Saved training analysis to {output_directory}")


if __name__ == "__main__":
    main()
