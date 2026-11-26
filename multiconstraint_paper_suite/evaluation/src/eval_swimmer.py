from pathlib import Path
import argparse
import sys

import numpy as np

from common import load_actor, load_json, load_normalizer, policy_action, reset_obs, resolve_checkpoint, save_seed_result, step_env, write_summary
from plotting import save_bundle


HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "config" / "runs.json"


def specs(repo, key, cfg):
    out = []
    for item in cfg["swimmer"][key]:
        directory = repo / "models" / item["namespace"] / f"run{item['run']}"
        out.append({"seed": item["seed"], "checkpoint": resolve_checkpoint(directory)})
    return out


def evaluate_one(checkpoint, seed, sigma, episodes, base_seed, device):
    from experiment_extensions.swimmer_multi_constraint_env import SwimmerMultiConstraintEnv
    env = SwimmerMultiConstraintEnv(sigma_viscosity=sigma)
    actor = load_actor(checkpoint, env, device)
    normalizer = load_normalizer(checkpoint)
    rewards = []
    max_costs = []
    total_costs = []
    for episode in range(episodes):
        obs = reset_obs(env, base_seed + episode)
        reward_sum = 0.0
        comp_max = np.zeros(2, dtype=float)
        comp_total = np.zeros(2, dtype=float)
        done = False
        steps = 0
        while not done and steps < 1000:
            action = policy_action(actor, obs, normalizer, device)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, info, returned_cost = step_env(env, action)
            components = np.asarray(info.get("multi_cost", returned_cost), dtype=float).reshape(-1)
            if components.shape != (2,):
                raise ValueError(f"Expected two Swimmer costs, got {components}")
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


def run_condition(repo, name, methods_spec, sigma, cfg, args):
    out = repo / "paper_evaluation" / "swimmer" / name
    methods = {}
    for label, key in methods_spec:
        methods[label] = []
        for spec in specs(repo, key, cfg):
            result = evaluate_one(spec["checkpoint"], spec["seed"], sigma, args.episodes, args.base_seed, args.device)
            save_seed_result(out, label, spec["seed"], result)
            methods[label].append({**spec, "result": result})
    threshold = float(cfg["swimmer"]["threshold"])
    write_summary(out, methods, threshold)
    save_bundle(
        methods,
        threshold,
        args.window,
        out,
        component_names=["C1", "C2"],
        totals=True,
        metadata={
            "environment": "Swimmer",
            "evaluation_sigma_viscosity": sigma,
            "episodes_per_training_seed": args.episodes,
            "evaluation_seed_start": args.base_seed,
            "evaluation_seed_end": args.base_seed + args.episodes - 1,
            "actor_evaluation": "deterministic_mean_action",
            "smoothing": f"moving average window {args.window}",
        },
    )
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--task", choices=("all", "np_ablation", "pr_ablation", "adaptation"), default="all")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=2301)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    cfg = load_json(CFG)
    sigma = float(cfg["swimmer"]["paper_eval_sigma"])
    jobs = {
        "np_ablation": ([('Multi Constraint', 'multi_np'), ('Combined Cost', 'combined_np')], 0.0),
        "pr_ablation": ([('Multi Constraint', 'multi_pr'), ('Combined Cost', 'combined_pr')], sigma),
        "adaptation": ([('Ours', 'multi_pr'), ('Surrogate Obj', 'multi_np')], sigma),
    }
    selected = jobs if args.task == "all" else {args.task: jobs[args.task]}
    for name, (method_spec, eval_sigma) in selected.items():
        run_condition(repo, name, method_spec, eval_sigma, cfg, args)


if __name__ == "__main__":
    main()
