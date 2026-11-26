#!/usr/bin/env bash
set -euo pipefail

cd /home/bannanakid/RCRL_on_RCMDP
mkdir -p logs

PYTHON="$(command -v python)"

echo "============================================================"
echo " SWIMMER ICLR TRAINING"
echo "============================================================"
echo "Repo:   $(pwd)"
echo "Python: $PYTHON"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"
echo "Start:  $(date -Is)"
echo

# ------------------------------------------------------------
# 0. PRE-FLIGHT
# ------------------------------------------------------------
"$PYTHON" iclr2027_safe_rl_extension_pack_v2/tools/preflight.py \
  --repo . \
  --require-swimmer

"$PYTHON" -m py_compile \
  code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py \
  code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
  code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
  code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py

"$PYTHON" - <<'PY'
import json
import math
from pathlib import Path

p = Path("experiment_extensions/swimmer_thresholds.json")
x = json.loads(p.read_text())

assert x["status"] == "CALIBRATED"
assert math.isclose(
    float(x["calibration"]["threshold_quantile"]),
    0.99,
    rel_tol=0.0,
    abs_tol=1e-12,
)
assert math.isclose(
    float(x["joint_speed_threshold"]),
    float(x["joint_speed_scale"]),
    rel_tol=0.0,
    abs_tol=1e-12,
)

print("SWIMMER CALIBRATION CHECK PASSED")
print("joint_speed_threshold =", x["joint_speed_threshold"])
print("joint_speed_scale     =", x["joint_speed_scale"])
PY

# ------------------------------------------------------------
# Fail-safe job runner.
#
# Explicitly checks Python's exit code even though output passes
# through tee. Therefore a traceback CANNOT silently continue.
# ------------------------------------------------------------
run_job () {
    LABEL="$1"
    LOGFILE="$2"
    shift 2

    echo
    echo "============================================================"
    echo "$LABEL"
    echo "Started: $(date -Is)"
    echo "Log: $LOGFILE"
    echo "============================================================"

    set +e
    "$@" 2>&1 | tee "$LOGFILE"
    PIPE_CODES=("${PIPESTATUS[@]}")
    set -e

    PY_STATUS="${PIPE_CODES[0]}"
    TEE_STATUS="${PIPE_CODES[1]}"

    if [ "$PY_STATUS" -ne 0 ]; then
        echo
        echo "ERROR: $LABEL FAILED"
        echo "Python exit code: $PY_STATUS"
        echo "Stopping the entire Swimmer queue."
        exit "$PY_STATUS"
    fi

    if [ "$TEE_STATUS" -ne 0 ]; then
        echo
        echo "ERROR: tee/logging failed for $LABEL"
        echo "Stopping the entire Swimmer queue."
        exit "$TEE_STATUS"
    fi

    echo
    echo "SUCCESS: $LABEL"
    echo "Finished: $(date -Is)"
}

# ------------------------------------------------------------
# Shared scientific settings
# ------------------------------------------------------------
COMMON=(
    --persistent_eps 0.1
    --beta 30000
    --K_epochs 5
    --max_train_steps 10000
    --evaluate_freq 100
    --warm_start_episode 150
    --lr_a 1e-3
    --lr_c 1e-3
    --lr_cost 1e-3
    --entropy_coef 0.001
    --batch_size 2048
    --mini_batch_size 128
)

# ------------------------------------------------------------
# 1. P+R SMOKE TESTS
#
# We already tested NP Multi + Combined.
# Verify perturbed paths before spending 10k episodes.
# ------------------------------------------------------------
echo
echo "Running P+R smoke tests before full training..."

run_job \
  "SMOKE: Swimmer Multi P+R" \
  "logs/swimmer_smoke_multi_pr.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
    --run 199812 \
    --seed 2 \
    --env SwimmerWithPosPerturbed \
    --persistent_eps 0.1 \
    --beta 30000 \
    --K_epochs 5 \
    --max_train_steps 1 \
    --evaluate_freq 100 \
    --warm_start_episode 150 \
    --lr_a 1e-3 \
    --lr_c 1e-3 \
    --lr_cost 1e-3 \
    --entropy_coef 0.001 \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7 \
    --batch_size 2048 \
    --mini_batch_size 128 \
    --cost_dim 2 \
    --cost_scale 1.0

run_job \
  "SMOKE: Swimmer Combined P+R" \
  "logs/swimmer_smoke_combined_pr.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py \
    --run 199912 \
    --seed 2 \
    --env SwimmerWithPosPerturbed \
    --persistent_eps 0.1 \
    --beta 30000 \
    --K_epochs 5 \
    --max_train_steps 1 \
    --evaluate_freq 100 \
    --warm_start_episode 150 \
    --lr_a 1e-3 \
    --lr_c 1e-3 \
    --lr_cost 1e-3 \
    --entropy_coef 0.001 \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7 \
    --batch_size 2048 \
    --mini_batch_size 128

echo
echo "P+R SMOKE TESTS PASSED."

# Remove only the temporary smoke-test run directories.
for ROOT in models data_train plot_data runs; do
    if [ -d "$ROOT" ]; then
        find "$ROOT" -type d \
          \( -name 'run199812' -o -name 'run199912' \) \
          -prune -exec rm -rf {} + 2>/dev/null || true
    fi
done

# ------------------------------------------------------------
# Fresh full-run IDs
# ------------------------------------------------------------
RUN_IDS=(11612 11712 11615 11715 11812 11912 11815 11915)

# Refuse to mix with an existing run carrying one of these IDs.
for RID in "${RUN_IDS[@]}"; do
    FOUND="$(
        find models data_train plot_data \
          -type d -name "run${RID}" \
          -print 2>/dev/null || true
    )"

    if [ -n "$FOUND" ]; then
        echo
        echo "ERROR: existing output found for run $RID:"
        echo "$FOUND"
        echo "Delete/rename it intentionally before launching."
        exit 1
    fi
done

echo
echo "FULL RUN DIRECTORY CHECK PASSED."
echo

# ============================================================
# FULL TRAINING
# ============================================================

# ------------------------------------------------------------
# 1/8 Multi NP, seed 2
# ------------------------------------------------------------
run_job \
  "1/8 Swimmer Multi NP | seed 2 | run11612" \
  "logs/swimmer_multi_np_seed2_run11612.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py \
    --run 11612 \
    --seed 2 \
    --env SwimmerWithPos \
    "${COMMON[@]}" \
    --weight_reg 0.0 \
    --sigma_viscosity 0.0 \
    --cost_dim 2 \
    --cost_scale 1.0

# ------------------------------------------------------------
# 2/8 Combined NP, seed 2
# ------------------------------------------------------------
run_job \
  "2/8 Swimmer Combined NP | seed 2 | run11712" \
  "logs/swimmer_combined_np_seed2_run11712.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
    --run 11712 \
    --seed 2 \
    --env SwimmerWithPos \
    "${COMMON[@]}" \
    --weight_reg 0.0 \
    --sigma_viscosity 0.0

# ------------------------------------------------------------
# 3/8 Multi NP, seed 5
# ------------------------------------------------------------
run_job \
  "3/8 Swimmer Multi NP | seed 5 | run11615" \
  "logs/swimmer_multi_np_seed5_run11615.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py \
    --run 11615 \
    --seed 5 \
    --env SwimmerWithPos \
    "${COMMON[@]}" \
    --weight_reg 0.0 \
    --sigma_viscosity 0.0 \
    --cost_dim 2 \
    --cost_scale 1.0

# ------------------------------------------------------------
# 4/8 Combined NP, seed 5
# ------------------------------------------------------------
run_job \
  "4/8 Swimmer Combined NP | seed 5 | run11715" \
  "logs/swimmer_combined_np_seed5_run11715.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
    --run 11715 \
    --seed 5 \
    --env SwimmerWithPos \
    "${COMMON[@]}" \
    --weight_reg 0.0 \
    --sigma_viscosity 0.0

# ------------------------------------------------------------
# 5/8 Multi P+R, seed 2
# ------------------------------------------------------------
run_job \
  "5/8 Swimmer Multi P+R | seed 2 | run11812" \
  "logs/swimmer_multi_pr_seed2_run11812.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
    --run 11812 \
    --seed 2 \
    --env SwimmerWithPosPerturbed \
    "${COMMON[@]}" \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7 \
    --cost_dim 2 \
    --cost_scale 1.0

# ------------------------------------------------------------
# 6/8 Combined P+R, seed 2
# ------------------------------------------------------------
run_job \
  "6/8 Swimmer Combined P+R | seed 2 | run11912" \
  "logs/swimmer_combined_pr_seed2_run11912.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py \
    --run 11912 \
    --seed 2 \
    --env SwimmerWithPosPerturbed \
    "${COMMON[@]}" \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7

# ------------------------------------------------------------
# 7/8 Multi P+R, seed 5
# ------------------------------------------------------------
run_job \
  "7/8 Swimmer Multi P+R | seed 5 | run11815" \
  "logs/swimmer_multi_pr_seed5_run11815.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
    --run 11815 \
    --seed 5 \
    --env SwimmerWithPosPerturbed \
    "${COMMON[@]}" \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7 \
    --cost_dim 2 \
    --cost_scale 1.0

# ------------------------------------------------------------
# 8/8 Combined P+R, seed 5
# ------------------------------------------------------------
run_job \
  "8/8 Swimmer Combined P+R | seed 5 | run11915" \
  "logs/swimmer_combined_pr_seed5_run11915.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py \
    --run 11915 \
    --seed 5 \
    --env SwimmerWithPosPerturbed \
    "${COMMON[@]}" \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7

echo
echo "============================================================"
echo " ALL 8 SWIMMER RUNS FINISHED SUCCESSFULLY"
echo " Finished: $(date -Is)"
echo "============================================================"

touch logs/SWIMMER_ALL_8_SUCCESS
