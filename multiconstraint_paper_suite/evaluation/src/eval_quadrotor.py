from pathlib import Path
import argparse
import sys

import numpy as np

from common import load_actor, load_json, load_normalizer, policy_action, reset_obs, resolve_checkpoint, save_seed_result, step_env, write_summary
from plotting import save_bundle


HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "config" / "runs.json"
PROFILES = HERE.parent / "profiles" / "quadrotor"


def specs(repo, key, cfg):
    out = []
    for item in cfg["quadrotor"][key]:
        directory = repo / "models" / item["namespace"] / f"run{item['run']}"
        try:
            checkpoint = resolve_checkpoint(directory)
        except FileNotFoundError:
            fallback = cfg["quadrotor"].get("fallbacks", {}).get(key, [])
            index = len(out)
            if index >= len(fallback):
                raise
            directory = repo / "models" / item["namespace"] / f"run{fallback[index]}"
            checkpoint = resolve_checkpoint(directory)
        out.append({"seed": item["seed"], "checkpoint": checkpoint})
    return out


def make_env(profile):
    from safe_control_gym.utils.configuration import ConfigFactory
    from safe_control_gym.utils.registration import make
    old = list(sys.argv)
    try:
        sys.argv = [old[0]]
        factory = ConfigFactory()
        factory.parser.set_defaults(overrides=[str(profile)])
        config = factory.merge()
    finally:
        sys.argv = old
    return make("quadrotor", **config.quadrotor_config)


def evaluate_one(checkpoint, seed, profile, cost_scale, episodes, base_seed, device):
    env = make_env(profile)
    actor = load_actor(checkpoint, env, device)
    normalizer = load_normalizer(checkpoint)
    rewards = []
    max_costs = []
    total_costs = []
    for episode in range(episodes):
        obs = reset_obs(env, base_seed + episode)
        reward_sum = 0.0
        comp_max = np.zeros(4, dtype=float)
        comp_total = np.zeros(4, dtype=float)
        done = False
        steps = 0
        while not done and steps < 2000:
            action = policy_action(actor, obs, normalizer, device)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, info, _ = step_env(env, action)
            if "constraint_values" not in info:
                raise KeyError(f"constraint_values missing from quadrotor info: {list(info)}")
            components = cost_scale * np.maximum(np.asarray(info["constraint_values"], dtype=float).reshape(-1), 0.0)
            if components.shape != (4,):
                raise ValueError(f"Expected four Quadrotor costs, got {components}")
            reward_sum += reward
            comp_total += components
            comp_max = np.maximum(comp_max, components)
            steps += 1
        rewards.append(reward_sum)
        max_costs.append(comp_max)
        total_costs.append(comp_total)
    env.close()
    return {
        "rewards": np.asarray(rewards),
        "component_max_costs": np.asarray(max_costs),
        "component_total_costs": np.asarray(total_costs),
    }


def run_condition(repo, name, methods_spec, profile_name, cfg, args, root=None):
    profile = PROFILES / f"{profile_name}.yaml"
    out = (root or (repo / "paper_evaluation" / "quadrotor")) / name
    methods = {}
    for label, key in methods_spec:
        methods[label] = []
        for spec in specs(repo, key, cfg):
            result = evaluate_one(
                spec["checkpoint"], spec["seed"], profile,
                float(cfg["quadrotor"]["cost_scale"]),
                args.episodes, args.base_seed, args.device,
            )
            save_seed_result(out, label, spec["seed"], result)
            methods[label].append({**spec, "result": result})
    threshold = float(cfg["quadrotor"]["threshold"])
    write_summary(out, methods, threshold)
    save_bundle(
        methods,
        threshold,
        args.window,
        out,
        component_names=["C1", "C2", "C3", "C4"],
        totals=True,
        metadata={
            "environment": "Quadrotor",
            "profile": profile_name,
            "profile_file": str(profile),
            "cost_scale": float(cfg["quadrotor"]["cost_scale"]),
            "episodes_per_training_seed": args.episodes,
            "evaluation_seed_start": args.base_seed,
            "evaluation_seed_end": args.base_seed + args.episodes - 1,
            "actor_evaluation": "deterministic_mean_action",
            "smoothing": f"moving average window {args.window}",
        },
    )
    print(out)


def run_robustness(repo, cfg, args):
    profiles = [
        "uniform_0p005", "uniform_0p01", "uniform_0p05",
        "white_noise_0p01", "white_noise_0p05",
        "horizontal_step_0p005", "horizontal_step_0p01",
    ]
    root = repo / "paper_evaluation" / "quadrotor" / "robustness"
    for profile in profiles:
        run_condition(
            repo,
            f"adaptation_{profile}",
            [("Ours", "multi_pr"), ("Surrogate Obj", "multi_np")],
            profile,
            cfg,
            args,
            root,
        )
        run_condition(
            repo,
            f"pr_ablation_{profile}",
            [("Multi Constraint", "multi_pr"), ("Combined Cost", "combined_pr")],
            profile,
            cfg,
            args,
            root,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--task", choices=("all", "np_ablation", "pr_ablation", "adaptation", "robustness"), default="all")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=2301)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    cfg = load_json(CFG)
    jobs = {
        "np_ablation": ([("Multi Constraint", "multi_np"), ("Combined Cost", "combined_np")], "nominal"),
        "pr_ablation": ([("Multi Constraint", "multi_pr"), ("Combined Cost", "combined_pr")], "uniform_0p005"),
        "adaptation": ([("Ours", "multi_pr"), ("Surrogate Obj", "multi_np")], "uniform_0p005"),
    }
    if args.task in ("all", "np_ablation", "pr_ablation", "adaptation"):
        selected = jobs if args.task == "all" else {args.task: jobs[args.task]}
        for name, (method_spec, profile) in selected.items():
            run_condition(repo, name, method_spec, profile, cfg, args)
    if args.task in ("all", "robustness"):
        run_robustness(repo, cfg, args)


if __name__ == "__main__":
    main()
