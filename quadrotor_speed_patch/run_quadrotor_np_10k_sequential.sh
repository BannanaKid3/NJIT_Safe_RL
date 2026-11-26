#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

mkdir -p logs

echo "=== Quadrotor NP 10k paired experiment ==="
echo "Repo: $REPO_ROOT"
echo "Python: $(command -v python)"
echo

# -------- Static sanity checks before expensive training --------
python -m py_compile \
  code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint.py \
  code_ipm_rcmdp_rcrl_max_quadrotor_combined.py

python - <<'PY'
from pathlib import Path
import yaml

for name in [
    "envs/env_configs/constrained_tracking_reset.yaml",
    "envs/env_configs/constrained_tracking_eval.yaml",
]:
    cfg = yaml.safe_load(Path(name).read_text())
    q = cfg["quadrotor_config"]

    if q.get("disturbances") not in (None, {}, []):
        raise SystemExit(f"REFUSING TO RUN NP: active disturbances found in {name}: {q['disturbances']}")

    cons = q.get("constraints", [])
    if len(cons) != 1:
        raise SystemExit(f"REFUSING TO RUN: expected exactly one bounded input constraint block in {name}, got {cons}")

    c = cons[0]
    if c.get("constrained_variable") != "input":
        raise SystemExit(f"REFUSING TO RUN: expected constrained_variable=input in {name}")

    if list(c.get("lower_bounds", [])) != [0.0, 0.0]:
        raise SystemExit(f"REFUSING TO RUN: unexpected lower_bounds in {name}: {c.get('lower_bounds')}")

    if list(c.get("upper_bounds", [])) != [0.12, 0.12]:
        raise SystemExit(f"REFUSING TO RUN: unexpected upper_bounds in {name}: {c.get('upper_bounds')}")

print("NP YAML sanity check PASSED: no disturbances, 2-D input bounds [0,0] to [0.12,0.12]")
PY

python - <<'PY'
import numpy as np
from experiment_extensions.quadrotor_costs import quadrotor_combined_cost_from_info

info = {"constraint_values": [-0.10, 0.20, 0.30, -0.40]}
combined, components = quadrotor_combined_cost_from_info(info, 100.0)

expected_components = [0.0, 20.0, 30.0, 0.0]
if not np.allclose(components, expected_components, rtol=0.0, atol=1e-5):
    raise SystemExit(f"Combined helper component test FAILED: {components}")
if not np.isclose(float(combined), 30.0, rtol=0.0, atol=1e-5):
    raise SystemExit(f"Combined helper max test FAILED: {combined}")

print("Combined cost helper sanity check PASSED: max of four clipped/scaled residuals")
PY

grep -q 'save_training_diagnostics = ((total_steps + 1) % 100 == 0)' \
  code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint.py
grep -q 'save_training_diagnostics = ((total_steps + 1) % 100 == 0)' \
  code_ipm_rcmdp_rcrl_max_quadrotor_combined.py

echo
echo "All pre-run checks passed."
echo

COMMON_ARGS=(
  --persistent_eps 0.5
  --beta 30000
  --K_epochs 5
  --max_train_steps 10000
  --warm_start_episode 1000
  --lr_cost 5e-4
  --lr_c 5e-4
  --lr_a 5e-4
  --entropy_coef 0.007
  --weight_reg 0.0
  --batch_size 1024
  --mini_batch_size 256
  --cost_scale 100.0
)

echo "=== 1/4 Multi NP seed 2, run9212 ==="
python code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint.py \
  --run 9212 --seed 2 "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/quad_multi_np_seed2_run9212.log \
&& \
echo "=== 2/4 Combined NP seed 2, run9312 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_quadrotor_combined.py \
  --run 9312 --seed 2 "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/quad_combined_np_seed2_run9312.log \
&& \
echo "=== 3/4 Multi NP seed 5, run9215 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint.py \
  --run 9215 --seed 5 "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/quad_multi_np_seed5_run9215.log \
&& \
echo "=== 4/4 Combined NP seed 5, run9315 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_quadrotor_combined.py \
  --run 9315 --seed 5 "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/quad_combined_np_seed5_run9315.log

echo
echo "ALL FOUR NP TRAINING RUNS FINISHED SUCCESSFULLY."
