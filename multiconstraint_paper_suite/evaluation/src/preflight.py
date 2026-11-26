from pathlib import Path
import argparse
import sys

from common import load_json, resolve_checkpoint
from eval_halfcheetah import scan


HERE = Path(__file__).resolve().parent
CFG = HERE.parent / "config" / "runs.json"


def check(value, text):
    print(("OK   " if value else "MISS ") + text)
    return bool(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--min-training-episodes", type=int, default=10000)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    cfg = load_json(CFG)
    ok = True
    for path in (
        repo / "experiment_extensions" / "swimmer_multi_constraint_env.py",
        repo / "experiment_extensions" / "swimmer_costs.py",
        repo / "experiment_extensions" / "swimmer_thresholds.json",
        repo / "envs" / "half_cheetah_gravity_perturbed_costs.py",
    ):
        ok &= check(path.exists(), str(path.relative_to(repo)))
    for env in ("quadrotor", "swimmer"):
        for key in ("multi_np", "combined_np", "multi_pr", "combined_pr"):
            for item in cfg[env][key]:
                directory = repo / "models" / item["namespace"] / f"run{item['run']}"
                found = True
                try:
                    checkpoint = resolve_checkpoint(directory)
                except FileNotFoundError:
                    found = False
                    checkpoint = directory
                ok &= check(found, f"{env} {key} seed {item['seed']}: {checkpoint}")
    for namespace in ("HalfCheetahMultiConstraintCost", "HalfCheetahCombinedMaxCost"):
        items = scan(repo, namespace, cfg, args.min_training_episodes)
        print(f"HC   {namespace}: {len(items)} completed paper runs")
        for item in sorted(items, key=lambda x: (x["robust"], x["train_sigma"] or 0, x["seed"])):
            mode = "P+R" if item["robust"] else "NP"
            sigma = "" if item["train_sigma"] is None else f" train_sigma={item['train_sigma']:g}"
            print(f"     {mode} seed={item['seed']} run={item['run']}{sigma} {item['checkpoint']}")
        ok &= bool(items)
    for profile in sorted((HERE.parent / "profiles" / "quadrotor").glob("*.yaml")):
        check(profile.exists(), f"profile {profile.name}")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
