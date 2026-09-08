import argparse
import json
import math
import os
from pathlib import Path


SCALAR_BASE_VERSION = "combined_max_wrapped90_jointp99_v1"
MULTI_BASE_VERSION = "multi_constraint_pitch_joint_v1"


def float_token(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Experiment parameters must be finite")
    text = format(value, ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def gravity_version(base_version, sigma_gravity, rho):
    return (
        f"{base_version}_gravity_step_normal_"
        f"sigma{float_token(sigma_gravity)}_rho{float_token(rho)}"
    )


def scalar_gravity_version(sigma_gravity, rho):
    return gravity_version(SCALAR_BASE_VERSION, sigma_gravity, rho)


def multi_gravity_version(sigma_gravity, rho):
    return gravity_version(MULTI_BASE_VERSION, sigma_gravity, rho)


def parse_int_list(value):
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return result


def parse_float_list(value):
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("At least one finite number is required")
    return result


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def add_training_arguments(parser, multi_constraint=False):
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sigma_gravity", type=float, required=True)
    parser.add_argument(
        "--rho",
        type=float,
        default=0.001,
        help=(
            "IPM ambiguity-radius coefficient. It is passed to the existing "
            "critic-norm term previously named weight_reg."
        ),
    )
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--evaluate_freq", type=float, default=100)
    parser.add_argument("--uncer_set", choices=("IPM",), default="IPM")
    parser.add_argument("--next_steps", type=int, default=2)
    parser.add_argument("--random_steps", type=int, default=int(25e3))
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--policy_dist", choices=("Gaussian",), default="Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--hidden_width", type=int, default=64)
    parser.add_argument("--lr_a", type=float, default=1e-3)
    parser.add_argument("--lr_c", type=float, default=5e-3)
    parser.add_argument("--lr_cost", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--persistent_eps", type=float, default=0.1)
    parser.add_argument("--K_epochs", type=int, default=5)
    parser.add_argument("--use_adv_norm", type=str_to_bool, default=True)
    parser.add_argument("--use_state_norm", type=str_to_bool, default=False)
    parser.add_argument("--use_reward_norm", type=str_to_bool, default=False)
    parser.add_argument("--use_reward_scaling", type=str_to_bool, default=False)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--use_lr_decay", type=str_to_bool, default=True)
    parser.add_argument("--use_grad_clip", type=str_to_bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=str_to_bool, default=True)
    parser.add_argument("--set_adam_eps", type=str_to_bool, default=True)
    parser.add_argument("--use_tanh", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--adaptive_alpha", type=str_to_bool, default=False)
    parser.add_argument("--baseline", type=int, default=9)
    parser.add_argument("--lambda_", type=int, default=50)
    parser.add_argument("--beta", type=float, default=3e4)
    parser.add_argument("--warm_start_flag", type=int, default=0)
    parser.add_argument("--warm_start_episode", type=int, default=500)
    parser.add_argument("--GAMMA", type=str, default="0")
    parser.add_argument("--gravity_std", type=float, default=0.5)
    if multi_constraint:
        parser.add_argument("--omega1", type=float, default=0.5)
        parser.add_argument("--omega2", type=float, default=0.5)
        parser.add_argument("--cost_dim", type=int, default=2)
        parser.add_argument("--cost_scale", type=float, default=1.0)
    return parser


def validate_training_arguments(args):
    if args.run < 1:
        raise ValueError("run must be at least 1")
    if not math.isfinite(args.sigma_gravity) or args.sigma_gravity < 0.0:
        raise ValueError("sigma_gravity must be a finite nonnegative number")
    if not math.isfinite(args.rho) or args.rho < 0.0:
        raise ValueError("rho must be a finite nonnegative number")
    if args.max_train_steps < 1:
        raise ValueError("max_train_steps must be at least 1")
    if args.evaluate_freq <= 0:
        raise ValueError("evaluate_freq must be positive")
    if args.batch_size < 1 or args.mini_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.mini_batch_size > args.batch_size:
        raise ValueError("mini_batch_size cannot exceed batch_size")
    args.weight_reg = float(args.rho)
    return args


def write_robust_config(
    repo_root,
    environment_name,
    version,
    args,
    method,
):
    output_directory = (
        Path(repo_root)
        / "data_train"
        / environment_name
        / version
        / f"run{args.run}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "environment": environment_name,
        "experiment_version": version,
        "run": int(args.run),
        "training_seed": int(args.seed),
        "sigma_gravity": float(args.sigma_gravity),
        "gravity_distribution": "Normal(base_gravity_z, sigma_gravity)",
        "gravity_resampled": "before_every_environment_step",
        "gravity_affects": ["next_state", "reward", "cost"],
        "rho": float(args.rho),
        "rho_implementation": (
            "Alias for the repository's existing weight_reg coefficient on the "
            "critic final-layer norm term; the underlying RCAC/IPM update is unchanged."
        ),
        "training_episodes": int(args.max_train_steps),
        "checkpoint_evaluation_frequency": int(args.evaluate_freq),
        "costs_unchanged": True,
    }
    path = output_directory / "robust_experiment_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def ensure_output_roots():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data_train", exist_ok=True)
    os.makedirs("plot_data", exist_ok=True)

