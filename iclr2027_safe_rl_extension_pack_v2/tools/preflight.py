from __future__ import annotations
from pathlib import Path
import argparse
import json
import py_compile
import sys

GENERATED = [
    "code_ipm_rcmdp_rcrl_max_quadrotor_combined.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py",
    "code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py",
    "code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py",
    "code_ipm_rcmdp_rcrl_max_swimmer_combined.py",
    "code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py",
]


def check_yaml_has_active_disturbance(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    # The generated profile YAMLs have an uncommented top-level-in-config disturbances block.
    return "\n  disturbances:\n" in text and "disturbance_func:" in text


def main():
    ap = argparse.ArgumentParser(description="Static preflight for the installed experiment overlay.")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--require-swimmer", action="store_true", help="Fail unless Swimmer calibration is complete")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    failures = []

    print("=== ICLR 2027 pipeline preflight ===")
    for rel in GENERATED:
        p = repo / rel
        if not p.exists():
            failures.append(f"missing generated trainer: {rel}")
            print("MISSING", rel)
            continue
        try:
            py_compile.compile(str(p), doraise=True)
            print("PY OK ", rel)
        except Exception as exc:
            failures.append(f"syntax error in {rel}: {exc}")
            print("PY FAIL", rel, exc)

    for rel in [
        "envs/env_configs/constrained_tracking_train_perturbed.yaml",
        "envs/env_configs/constrained_tracking_eval_perturbed.yaml",
    ]:
        p = repo / rel
        if not p.exists():
            failures.append(f"missing perturbed YAML: {rel}")
            print("YAML MISSING", rel)
        elif not check_yaml_has_active_disturbance(p):
            failures.append(f"no active disturbance found in {rel}")
            print("YAML FAIL", rel)
        else:
            print("YAML OK ", rel)

    active = repo / "envs/env_configs/ACTIVE_PERTURBATION_PROFILE.txt"
    if active.exists():
        print("QUAD PROFILE:", active.read_text(encoding="utf-8").strip())
    else:
        print("QUAD PROFILE: marker missing")

    thresholds = repo / "experiment_extensions/swimmer_thresholds.json"
    swimmer_ready = False
    if thresholds.exists():
        try:
            data = json.loads(thresholds.read_text(encoding="utf-8"))
            swimmer_ready = (
                data.get("status") == "CALIBRATED"
                and data.get("joint_speed_threshold") is not None
                and data.get("joint_speed_scale") is not None
            )
            if swimmer_ready:
                print("SWIMMER READY:", thresholds)
                print("  joint_speed_threshold =", data["joint_speed_threshold"])
                print("  joint_speed_scale     =", data["joint_speed_scale"])
            else:
                print("SWIMMER NOT CALIBRATED YET. This is expected before step 5.")
        except Exception as exc:
            print("SWIMMER CALIBRATION FILE INVALID:", exc)
    else:
        print("SWIMMER CALIBRATION FILE MISSING")

    if args.require_swimmer and not swimmer_ready:
        failures.append("Swimmer calibration is required for this launch")

    if failures:
        print("\nPRE-FLIGHT FAILED")
        for item in failures:
            print(" -", item)
        sys.exit(2)
    print("\nPRE-FLIGHT PASSED for Quadrotor and static code generation.")
    if not swimmer_ready:
        print("Run the Swimmer calibration step before starting any generated Swimmer trainer.")


if __name__ == "__main__":
    main()
