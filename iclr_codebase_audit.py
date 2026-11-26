#!/usr/bin/env python3
"""
Read-only ICLR experiment codebase audit for RCRL_on_RCMDP.

This script does NOT edit any repository file. It checks the live Ubuntu clone
for the experiment assumptions discussed for the ICLR multi-constraint study.

Exit codes:
  0 = no hard failures (warnings may remain)
  2 = one or more hard failures
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json
import py_compile
import re
import subprocess
import sys
import tempfile
import yaml
import numpy as np

QUAD_PATCH_TARGETS = [
    "code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_combined.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py",
]

GENERATED = [
    "code_ipm_rcmdp_rcrl_max_quadrotor_combined.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py",
    "code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py",
    "code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py",
    "code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py",
    "code_ipm_rcmdp_rcrl_max_swimmer_combined.py",
    "code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py",
]

HALFCHEETAH_REQUIRED = [
    "halfcheetah_safe_rl_pipeline/02_cost_definitions/envs/half_cheetah_multi_constraint_cost.py",
    "halfcheetah_safe_rl_pipeline/02_cost_definitions/envs/half_cheetah_combined_max_cost.py",
    "halfcheetah_safe_rl_pipeline/03_nominal_training/train_multi_constraint.py",
    "halfcheetah_safe_rl_pipeline/03_nominal_training/train_scalar_combined.py",
    "halfcheetah_safe_rl_pipeline/06_gravity_robustness/half_cheetah_gravity_perturbed_costs.py",
    "halfcheetah_safe_rl_pipeline/06_gravity_robustness/train_multi_gravity_robust.py",
    "halfcheetah_safe_rl_pipeline/06_gravity_robustness/train_scalar_gravity_robust.py",
    "halfcheetah_safe_rl_pipeline/06_gravity_robustness/evaluate_gravity_robustness.py",
]

failures = []
warnings = []
passes = []

def PASS(msg):
    passes.append(msg)
    print("PASS:", msg)

def WARN(msg):
    warnings.append(msg)
    print("WARN:", msg)

def FAIL(msg):
    failures.append(msg)
    print("FAIL:", msg)

def git(repo: Path, *args):
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def parse_args_defaults(path: Path):
    """Best-effort extraction of parser.add_argument defaults from source AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        names = []
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                names.append(a.value)
        if not names:
            continue
        default_node = None
        for kw in node.keywords:
            if kw.arg == "default":
                default_node = kw.value
                break
        if default_node is None:
            continue
        try:
            default = ast.literal_eval(default_node)
        except Exception:
            continue
        for name in names:
            out[name] = default
    return out

def active_disturbance(q):
    d = q.get("disturbances")
    return d not in (None, {}, [])

def check_quad_yaml(repo: Path):
    print("\n=== QUADROTOR YAML ===")
    for rel in [
        "envs/env_configs/constrained_tracking_reset.yaml",
        "envs/env_configs/constrained_tracking_eval.yaml",
    ]:
        p = repo / rel
        if not p.exists():
            FAIL(f"missing nominal YAML {rel}")
            continue
        q = yaml.safe_load(p.read_text(encoding="utf-8"))["quadrotor_config"]
        if active_disturbance(q):
            FAIL(f"NP YAML has active disturbance: {rel}: {q.get('disturbances')}")
        else:
            PASS(f"NP YAML has no active disturbance: {rel}")
        cons = q.get("constraints", [])
        if (
            len(cons) == 1
            and cons[0].get("constrained_variable") == "input"
            and list(cons[0].get("lower_bounds", [])) == [0.0, 0.0]
            and list(cons[0].get("upper_bounds", [])) == [0.12, 0.12]
        ):
            PASS(f"2-D input bounds are [0,0] to [0.12,0.12]: {rel}")
        else:
            FAIL(f"unexpected Quad constraint block in {rel}: {cons}")

    marker = repo / "envs/env_configs/ACTIVE_PERTURBATION_PROFILE.txt"
    if marker.exists():
        profile = marker.read_text(encoding="utf-8").strip()
        if profile == "uniform_0p005":
            PASS("active paper-facing Quad P profile marker is uniform_0p005")
        else:
            WARN(f"active Quad P profile is {profile!r}; historical Shilpa P profile expected uniform_0p005")
    else:
        WARN("ACTIVE_PERTURBATION_PROFILE.txt missing")

    for rel in [
        "envs/env_configs/constrained_tracking_train_perturbed.yaml",
        "envs/env_configs/constrained_tracking_eval_perturbed.yaml",
    ]:
        p = repo / rel
        if not p.exists():
            FAIL(f"missing perturbed YAML {rel}")
            continue
        q = yaml.safe_load(p.read_text(encoding="utf-8"))["quadrotor_config"]
        dyn = (q.get("disturbances") or {}).get("dynamics", [])
        if (
            len(dyn) == 1
            and dyn[0].get("disturbance_func") == "uniform"
            and list(dyn[0].get("low", [])) == [-0.005, -0.005]
            and list(dyn[0].get("high", [])) == [0.005, 0.005]
        ):
            PASS(f"perturbed YAML uses uniform dynamics ±0.005: {rel}")
        else:
            WARN(f"{rel} is not uniform_0p005: {dyn}")

def check_generated(repo: Path):
    print("\n=== GENERATED TRAINERS ===")
    for rel in GENERATED:
        p = repo / rel
        if not p.exists():
            FAIL(f"missing generated trainer {rel}")
            continue
        try:
            py_compile.compile(str(p), doraise=True)
            PASS(f"py_compile {rel}")
        except Exception as e:
            FAIL(f"syntax error {rel}: {e}")

    combined = repo / "code_ipm_rcmdp_rcrl_max_quadrotor_combined.py"
    if combined.exists():
        text = combined.read_text(encoding="utf-8")
        n = text.count("quadrotor_combined_cost_from_info(info, args.cost_scale)")
        if n == 2:
            PASS("Quad Combined uses helper in exactly train + evaluation paths")
        else:
            FAIL(f"Quad Combined helper call count expected 2, got {n}")

    helper = repo / "experiment_extensions/quadrotor_costs.py"
    if helper.exists():
        sys.path.insert(0, str(repo))
        try:
            from experiment_extensions.quadrotor_costs import quadrotor_combined_cost_from_info
            c, parts = quadrotor_combined_cost_from_info(
                {"constraint_values": [-0.1, 0.02, 0.03, 0.04]}, 100.0
            )
            if np.allclose(parts, [0.0, 2.0, 3.0, 4.0]) and np.isclose(c, 4.0):
                PASS("Quad Combined objective is max of all four clipped/scaled residuals")
            else:
                FAIL(f"Quad Combined helper unexpected result: scalar={c}, parts={parts}")
        except Exception as e:
            FAIL(f"Quad Combined helper test failed: {e}")
    else:
        FAIL("missing experiment_extensions/quadrotor_costs.py")

def check_speed_patch(repo: Path):
    print("\n=== SPEED-ONLY PATCH ===")
    for rel in QUAD_PATCH_TARGETS:
        p = repo / rel
        if not p.exists():
            FAIL(f"missing speed-patch target {rel}")
            continue
        text = p.read_text(encoding="utf-8")
        if "save_training_diagnostics = ((total_steps + 1) % 100 == 0)" in text:
            PASS(f"100-episode diagnostic gating present: {rel}")
        else:
            FAIL(f"diagnostic gating missing: {rel}")

        active_render = [
            i + 1 for i, line in enumerate(text.splitlines())
            if re.match(r"^[ \t]*env\.render\(\)\s*(#.*)?$", line)
        ]
        if active_render:
            FAIL(f"active env.render() remains in {rel} at lines {active_render}")
        else:
            PASS(f"no active standalone env.render(): {rel}")

        backup = p.with_name(p.name + ".pre_io_opt.bak")
        if backup.exists():
            PASS(f"pre-I/O optimization backup exists: {backup.name}")
        else:
            WARN(f"backup missing for {rel}; cannot independently roll back speed patch")

def check_runner(repo: Path):
    print("\n=== NP BASH RUNNER ===")
    p = repo / "quadrotor_speed_patch/run_quadrotor_np_10k_sequential.sh"
    if not p.exists():
        WARN("NP runner not found at quadrotor_speed_patch/run_quadrotor_np_10k_sequential.sh")
        return
    text = p.read_text(encoding="utf-8")
    try:
        subprocess.run(["bash", "-n", str(p)], check=True, capture_output=True, text=True)
        PASS("bash -n NP runner")
    except subprocess.CalledProcessError as e:
        FAIL(f"NP runner syntax error: {e.stderr}")

    required = [
        ("--run 211 --seed 2", "Multi seed2 run211"),
        ("--run 201 --seed 2", "Combined seed2 run201"),
        ("--run 215 --seed 5", "Multi seed5 run215"),
        ("--run 205 --seed 5", "Combined seed5 run205"),
        ("--max_train_steps 10000", "10k training budget"),
        ("set -euo pipefail", "fail-fast + pipefail"),
    ]
    for token, desc in required:
        if token in text:
            PASS(f"runner contains {desc}")
        else:
            FAIL(f"runner missing {desc}")

    # The intentional command chain uses &&. We flag common background forms.
    if re.search(r"\)\s*&\s*(?:\n|$)", text) or re.search(r"\bpython\b[^\n]*\s&\s*(?:\n|$)", text):
        WARN("runner appears to contain a background '&' job; inspect before running")
    else:
        PASS("runner has no obvious background-parallel launch; runs are sequential")
    if text.count("&&") >= 3:
        PASS("runner chains jobs with && so next job starts only after prior success")
    else:
        FAIL("runner does not contain expected && chain")

def check_halfcheetah(repo: Path):
    print("\n=== HALFCHEETAH PIPELINE ===")
    for rel in HALFCHEETAH_REQUIRED:
        p = repo / rel
        if p.exists():
            PASS(f"present {rel}")
        else:
            FAIL(f"missing {rel}")

    multi = repo / "halfcheetah_safe_rl_pipeline/03_nominal_training/train_multi_constraint.py"
    if multi.exists():
        try:
            d = parse_args_defaults(multi)
            if d.get("--cost_dim") == 2:
                PASS("HalfCheetah Multi cost_dim default is 2")
            else:
                WARN(f"HalfCheetah Multi cost_dim default={d.get('--cost_dim')}")
            if d.get("--persistent_eps") == 0.1:
                PASS("HalfCheetah Multi persistent_eps default is 0.1")
            if d.get("--evaluate_freq") == 100.0:
                PASS("HalfCheetah Multi evaluate_freq default is 100")
        except Exception as e:
            WARN(f"could not parse HalfCheetah Multi defaults: {e}")

    scalar = repo / "halfcheetah_safe_rl_pipeline/03_nominal_training/train_scalar_combined.py"
    if scalar.exists():
        txt = scalar.read_text(encoding="utf-8")
        if 'if args.seed != 5' in txt:
            WARN("HalfCheetah Combined nominal trainer currently hard-locks the official run to seed 5; a second-seed paper rerun needs this generalized or an already-existing matched run")
        if 'if args.max_train_steps != 2000' in txt:
            WARN("HalfCheetah Combined nominal trainer currently hard-locks the official run to 2000 episodes")

    plotter = repo / "halfcheetah_safe_rl_pipeline/06_gravity_robustness/plot_gravity_training.py"
    if plotter.exists():
        txt = plotter.read_text(encoding="utf-8")
        if 'default=100)' in txt and '±1 SD' in txt:
            WARN("existing HalfCheetah robustness plotter uses smoothing=100 and ±1 SD, not the current paper convention of first-10 raw + window-10 and ±1.5 SD")

    evaluator = repo / "halfcheetah_safe_rl_pipeline/06_gravity_robustness/evaluate_gravity_robustness.py"
    if evaluator.exists():
        txt = evaluator.read_text(encoding="utf-8")
        if 'parser.add_argument("--episodes", type=int, default=100)' in txt:
            PASS("HalfCheetah robustness evaluator defaults to 100 episodes")
        if 'default=[0.0, 1.0, 2.0]' in txt:
            WARN("HalfCheetah evaluator default eval sigmas omit 1.5; pass --eval_sigmas 1.5 for Shilpa's requested perturbed inference")

def check_swimmer(repo: Path):
    print("\n=== SWIMMER THIRD ENV ===")
    th = repo / "experiment_extensions/swimmer_thresholds.json"
    if not th.exists():
        WARN("Swimmer threshold file missing; Swimmer generated runs must not start")
        return
    try:
        data = json.loads(th.read_text(encoding="utf-8"))
    except Exception as e:
        FAIL(f"invalid Swimmer threshold JSON: {e}")
        return
    if (
        data.get("status") == "CALIBRATED"
        and isinstance(data.get("joint_speed_threshold"), (int, float))
        and isinstance(data.get("joint_speed_scale"), (int, float))
    ):
        PASS("Swimmer joint-speed calibration is complete")
    else:
        WARN("Swimmer joint-speed calibration is not complete yet; this is an intentional blocker before Swimmer training")

def check_package_logic(repo: Path):
    print("\n=== EXTENSION PACKAGE / PAPER PLOTTING ===")
    pkg = repo / "iclr2027_safe_rl_extension_pack_v2"
    if not pkg.exists():
        WARN("extension pack directory not found")
        return
    plotter = pkg / "paper_plotting/paper_style_plots.py"
    if plotter.exists():
        text = plotter.read_text(encoding="utf-8")
        required = [
            "1.5 * sd",
            "window=10",
            "prefix=10",
            'raise ValueError("Paper aggregation in this project expects exactly two seeds")',
            "np.max(c, axis=1)",
        ]
        if all(x in text for x in required):
            PASS("paper plotter encodes two seeds, ±1.5 SD, first-10 raw/window-10, and worst/MAX vector cost")
        else:
            WARN("paper plotter does not match all expected plotting conventions")
    else:
        WARN("paper_style_plots.py missing")

    setup = pkg / "setup_pipeline.py"
    if setup.exists():
        txt = setup.read_text(encoding="utf-8")
        if "default='white_noise_0p01'" in txt:
            WARN("setup_pipeline.py still defaults to white_noise_0p01; always pass --quad-profile uniform_0p005 for the Shilpa-matched historical P setup")

def check_final_inference_gap(repo: Path):
    print("\n=== FINAL-INFERENCE READINESS ===")
    analysis = repo / "iclr2027_safe_rl_extension_pack_v2/analysis/analyze_two_seed_method.py"
    if analysis.exists():
        txt = analysis.read_text(encoding="utf-8")
        if "feasible in both seeds" in txt:
            WARN("current two-seed analysis selects one common checkpoint index across seeds; for final inference, select the best feasible checkpoint independently per training seed unless the paper explicitly fixes one shared checkpoint index")
    WARN("verify a dedicated 100-episode final evaluator for the new Quad Multi/Combined and Swimmer Multi/Combined checkpoints before calling the paper pipeline complete")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    print("=== ICLR CODEBASE READ-ONLY AUDIT ===")
    print("repo:", repo)

    if not (repo / ".git").exists():
        FAIL(".git not found; run from the RCRL_on_RCMDP repository root")
    else:
        rc, branch, err = git(repo, "branch", "--show-current")
        if rc == 0:
            print("branch:", branch)
        rc, status, err = git(repo, "status", "--short")
        if rc == 0:
            if status:
                WARN("working tree has uncommitted changes/generated files; freeze/commit the exact paper code before final paper runs")
                print(status[:6000])
            else:
                PASS("working tree clean")

    check_quad_yaml(repo)
    check_generated(repo)
    check_speed_patch(repo)
    check_runner(repo)
    check_halfcheetah(repo)
    check_swimmer(repo)
    check_package_logic(repo)
    check_final_inference_gap(repo)

    print("\n=== SUMMARY ===")
    print(f"PASS={len(passes)} WARN={len(warnings)} FAIL={len(failures)}")
    if failures:
        print("\nHARD FAILURES:")
        for x in failures:
            print(" -", x)
    if warnings:
        print("\nWARNINGS / PAPER-READINESS ITEMS:")
        for x in warnings:
            print(" -", x)

    if failures:
        return 2
    print("\nNo hard failure detected. Warnings still need scientific/reproducibility review before final paper claims.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
