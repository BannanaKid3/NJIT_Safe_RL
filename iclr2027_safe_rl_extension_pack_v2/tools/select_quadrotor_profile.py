from __future__ import annotations
from pathlib import Path
import argparse
import shutil

PACK = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Switch the explicit Quadrotor perturbed train/eval YAML pair.")
    ap.add_argument("--repo", default=".", help="Path to the RCRL_on_RCMDP repository")
    ap.add_argument("--profile", required=True, help="Profile folder name under env_profiles/quadrotor")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    profile = PACK / "env_profiles" / "quadrotor" / args.profile
    if not profile.is_dir():
        choices = sorted(p.name for p in (PACK / "env_profiles" / "quadrotor").iterdir() if p.is_dir())
        raise SystemExit(f"Unknown profile {args.profile!r}. Choices: {choices}")

    out = repo / "envs" / "env_configs"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile / "train.yaml", out / "constrained_tracking_train_perturbed.yaml")
    shutil.copy2(profile / "eval.yaml", out / "constrained_tracking_eval_perturbed.yaml")
    (out / "ACTIVE_PERTURBATION_PROFILE.txt").write_text(args.profile + "\n", encoding="utf-8")
    print(f"Active Quadrotor perturbation profile: {args.profile}")
    print(out / "constrained_tracking_train_perturbed.yaml")
    print(out / "constrained_tracking_eval_perturbed.yaml")


if __name__ == "__main__":
    main()
