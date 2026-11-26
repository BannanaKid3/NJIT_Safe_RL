from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np

from common import load_actor, load_json, load_normalizer, policy_action, reset_obs, resolve_checkpoint, save_seed_result, step_env, write_summary
from plotting import save_bundle


HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "config" / "runs.json"


def decode_token(text):
    return float(text.replace("m", "-").replace("p", "."))


def scan(repo, namespace, cfg, min_episodes):
    root = repo / "models" / namespace
    out = []
    if not root.exists():
        return out
    seed_map = {int(k): int(v) for k, v in cfg["halfcheetah"]["nominal_seed_by_run"].items()}
    for run_dir in sorted(root.rglob("run*")):
        if not run_dir.is_dir() or not re.fullmatch(r"run\d+", run_dir.name):
            continue
        run = int(run_dir.name[3:])
        try:
            checkpoint = resolve_checkpoint(run_dir)
        except FileNotFoundError:
            continue
        rel = run_dir.relative_to(repo / "models")
        plot_dir = repo / "plot_data" / rel
        reward_path = plot_dir / "episode_rewards.npy"
        if not reward_path.exists():
            continue
        try:
            episode_count = len(np.load(reward_path, mmap_mode="r"))
        except Exception:
            continue
        if episode_count < min_episodes:
            continue
        data_dir = repo / "data_train" / rel
        meta = {}
        for name in ("robust_experiment_config.json", "training_metadata.json"):
            path = data_dir / name
            if path.exists():
                try:
                    meta.update(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    pass
        version = "/".join(rel.parts[1:-1])
        robust = "gravity_step_normal" in version or "sigma_gravity" in meta
        sigma = meta.get("sigma_gravity")
        if sigma is None and robust:
            match = re.search(r"sigma([0-9mp]+)_rho", version)
            sigma = decode_token(match.group(1)) if match else None
        seed = meta.get("training_seed")
        if seed is None:
            seed = seed_map.get(run)
        if seed is None:
            continue
        out.append({
            "seed": int(seed),
            "run": run,
            "checkpoint": checkpoint,
            "version": version,
            "robust": robust,
            "train_sigma": None if sigma is None else float(sigma),
            "episodes": episode_count,
        })
    unique = {}
    for item in out:
        key = (item["seed"], item["robust"], item["train_sigma"])
        if key not in unique or item["episodes"] > unique[key]["episodes"]:
            unique[key] = item
    return list(unique.values())


def make_env(repo, sigma):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "halfcheetah_safe_rl_pipeline" / "06_gravity_robustness"))
    try:
        from envs.half_cheetah_gravity_perturbed_costs import HalfCheetahCombinedMaxCostGravityPerturbed
    except ImportError:
        from half_cheetah_gravity_perturbed_costs import HalfCheetahCombinedMaxCostGravityPerturbed
    return HalfCheetahCombinedMaxCostGravityPerturbed(sigma_gravity=sigma)


def evaluate_one(repo, checkpoint, seed, sigma, episodes, base_seed, device):
    env = make_env(repo, sigma)
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
        max_steps = int(getattr(env, "max_steps", 1000))
        while not done and steps < max_steps:
            action = policy_action(actor, obs, normalizer, device)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, info, _ = step_env(env, action)
            components = np.asarray([info["pitch_cost"], info["joint_speed_cost"]], dtype=float)
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


def same_seeds(a, b):
    aa = {item["seed"]: item for item in a}
    bb = {item["seed"]: item for item in b}
    common = sorted(set(aa) & set(bb))
    if common:
        return [aa[x] for x in common], [bb[x] for x in common]
    return a, b


def run_condition(repo, name, method_items, eval_sigma, cfg, args, metadata=None):
    out = repo / "paper_evaluation" / "halfcheetah" / name
    methods = {}
    for label, items in method_items:
        methods[label] = []
        for spec in items:
            result = evaluate_one(repo, spec["checkpoint"], spec["seed"], eval_sigma, args.episodes, args.base_seed, args.device)
            save_seed_result(out, label, spec["seed"], result)
            methods[label].append({**spec, "result": result})
    threshold = float(cfg["halfcheetah"]["threshold"])
    write_summary(out, methods, threshold)
    meta = {
        "environment": "HalfCheetah",
        "evaluation_gravity_sigma": eval_sigma,
        "episodes_per_training_seed": args.episodes,
        "evaluation_seed_start": args.base_seed,
        "evaluation_seed_end": args.base_seed + args.episodes - 1,
        "actor_evaluation": "deterministic_mean_action",
        "smoothing": f"moving average window {args.window}",
    }
    if metadata:
        meta.update(metadata)
    save_bundle(methods, threshold, args.window, out, component_names=["C1", "C2"], totals=True, metadata=meta)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--task", choices=("all", "np_ablation", "pr_ablation", "adaptation"), default="all")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=2301)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-training-episodes", type=int, default=10000)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    cfg = load_json(CFG)
    multi = scan(repo, "HalfCheetahMultiConstraintCost", cfg, args.min_training_episodes)
    combined = scan(repo, "HalfCheetahCombinedMaxCost", cfg, args.min_training_episodes)
    m_np = [x for x in multi if not x["robust"]]
    c_np = [x for x in combined if not x["robust"]]
    if args.task in ("all", "np_ablation"):
        if not m_np or not c_np:
            raise FileNotFoundError("Completed HalfCheetah nominal Multi/Combined checkpoints were not both found")
        a, b = same_seeds(m_np, c_np)
        run_condition(repo, "np_ablation", [("Multi Constraint", a), ("Combined Cost", b)], 0.0, cfg, args)
    paper_sigma = float(cfg["halfcheetah"]["paper_eval_sigma"])
    pr_sigmas = sorted(set(x["train_sigma"] for x in multi + combined if x["robust"] and x["train_sigma"] is not None))
    for train_sigma in pr_sigmas:
        m_pr = [x for x in multi if x["robust"] and x["train_sigma"] == train_sigma]
        c_pr = [x for x in combined if x["robust"] and x["train_sigma"] == train_sigma]
        token = str(train_sigma).replace(".", "p")
        if args.task in ("all", "pr_ablation") and m_pr and c_pr:
            a, b = same_seeds(m_pr, c_pr)
            run_condition(
                repo,
                f"pr_ablation_train_sigma_{token}",
                [("Multi Constraint", a), ("Combined Cost", b)],
                paper_sigma,
                cfg,
                args,
                {"training_gravity_sigma": train_sigma},
            )
        if args.task in ("all", "adaptation") and m_pr and m_np:
            a, b = same_seeds(m_pr, m_np)
            run_condition(
                repo,
                f"adaptation_train_sigma_{token}",
                [("Ours", a), ("Surrogate Obj", b)],
                paper_sigma,
                cfg,
                args,
                {"training_gravity_sigma": train_sigma},
            )
    if args.task in ("pr_ablation", "adaptation") and not pr_sigmas:
        raise FileNotFoundError("No completed 10k HalfCheetah robust checkpoints were found")


if __name__ == "__main__":
    main()
