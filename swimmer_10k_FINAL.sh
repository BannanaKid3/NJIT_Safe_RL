#!/usr/bin/env bash
set -euo pipefail

cd /home/bannanakid/RCRL_on_RCMDP

mkdir -p logs

PYTHON="$(command -v python)"

echo "============================================================"
echo " SWIMMER FINAL 10K PAIRED TRAINING"
echo "============================================================"
echo "Repo:   $(pwd)"
echo "Python: $PYTHON"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"
echo "Start:  $(date -Is)"
echo

# ------------------------------------------------------------
# Preflight before spending compute
# ------------------------------------------------------------

"$PYTHON" iclr2027_safe_rl_extension_pack_v2/tools/preflight.py \
  --repo . \
  --require-swimmer

"$PYTHON" -m py_compile \
  code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py \
  code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
  code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
  code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py


# ------------------------------------------------------------
# Refuse stale paper output directories
# ------------------------------------------------------------

check_fresh () {
    TAG="$1"
    RID="$2"

    for ROOT in models data_train plot_data; do
        D="$ROOT/$TAG/run$RID"

        if [ -e "$D" ]; then
            echo "ERROR: existing output directory:"
            echo "$D"
            echo "Stopping rather than mixing old/new results."
            exit 1
        fi
    done
}

check_fresh SwimmerMulti                    11612
check_fresh SwimmerCombined                 11712
check_fresh SwimmerMulti                    11615
check_fresh SwimmerCombined                 11715
check_fresh SwimmerMultiPerturbedRobust     11812
check_fresh SwimmerCombinedPerturbedRobust  11912
check_fresh SwimmerMultiPerturbedRobust     11815
check_fresh SwimmerCombinedPerturbedRobust  11915

echo "ALL 8 OUTPUT DIRECTORY CHECKS PASSED"


# ------------------------------------------------------------
# Fail-safe sequential job function
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

    PIPE_RESULTS=("${PIPESTATUS[@]}")

    set -e

    PY_RC="${PIPE_RESULTS[0]}"
    TEE_RC="${PIPE_RESULTS[1]}"

    if [ "$PY_RC" -ne 0 ]; then
        echo
        echo "ERROR: $LABEL FAILED"
        echo "Python exit code: $PY_RC"
        echo "NO LATER TRAINING JOBS WILL START."
        exit "$PY_RC"
    fi

    if [ "$TEE_RC" -ne 0 ]; then
        echo
        echo "ERROR: logging failed for $LABEL"
        echo "tee exit code: $TEE_RC"
        exit "$TEE_RC"
    fi

    echo
    echo "SUCCESS: $LABEL"
    echo "Finished: $(date -Is)"
}


# ------------------------------------------------------------
# Same 10K budget for ALL new Swimmer methods
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


# ============================================================
# 1/8 MULTI NP SEED 2
# ============================================================

run_job \
  "1/8 Swimmer Multi NP | seed 2 | run11612 | 10k" \
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


# ============================================================
# 2/8 COMBINED NP SEED 2
# ============================================================

run_job \
  "2/8 Swimmer Combined NP | seed 2 | run11712 | 10k" \
  "logs/swimmer_combined_np_seed2_run11712.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
    --run 11712 \
    --seed 2 \
    --env SwimmerWithPos \
    "${COMMON[@]}" \
    --weight_reg 0.0 \
    --sigma_viscosity 0.0


# ============================================================
# 3/8 MULTI NP SEED 5
# ============================================================

run_job \
  "3/8 Swimmer Multi NP | seed 5 | run11615 | 10k" \
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


# ============================================================
# 4/8 COMBINED NP SEED 5
# ============================================================

run_job \
  "4/8 Swimmer Combined NP | seed 5 | run11715 | 10k" \
  "logs/swimmer_combined_np_seed5_run11715.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
    --run 11715 \
    --seed 5 \
    --env SwimmerWithPos \
    "${COMMON[@]}" \
    --weight_reg 0.0 \
    --sigma_viscosity 0.0


# ============================================================
# 5/8 MULTI P+R SEED 2
#
# IMPORTANT:
# The generated Multi trainer binds SwimmerWithPos to
# SwimmerMultiConstraintEnv.
# Perturbation comes from sigma_viscosity=0.7.
# ============================================================

run_job \
  "5/8 Swimmer Multi P+R | seed 2 | run11812 | 10k" \
  "logs/swimmer_multi_pr_seed2_run11812.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
    --run 11812 \
    --seed 2 \
    --env SwimmerWithPos \
    "${COMMON[@]}" \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7 \
    --cost_dim 2 \
    --cost_scale 1.0


# ============================================================
# 6/8 COMBINED P+R SEED 2
# ============================================================

run_job \
  "6/8 Swimmer Combined P+R | seed 2 | run11912 | 10k" \
  "logs/swimmer_combined_pr_seed2_run11912.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py \
    --run 11912 \
    --seed 2 \
    --env SwimmerWithPosPerturbed \
    "${COMMON[@]}" \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7


# ============================================================
# 7/8 MULTI P+R SEED 5
# ============================================================

run_job \
  "7/8 Swimmer Multi P+R | seed 5 | run11815 | 10k" \
  "logs/swimmer_multi_pr_seed5_run11815.log" \
  "$PYTHON" code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
    --run 11815 \
    --seed 5 \
    --env SwimmerWithPos \
    "${COMMON[@]}" \
    --weight_reg 0.001 \
    --sigma_viscosity 0.7 \
    --cost_dim 2 \
    --cost_scale 1.0


# ============================================================
# 8/8 COMBINED P+R SEED 5
# ============================================================

run_job \
  "8/8 Swimmer Combined P+R | seed 5 | run11915 | 10k" \
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
echo " ALL 8 SWIMMER 10K RUNS FINISHED SUCCESSFULLY"
echo " Finished: $(date -Is)"
echo "============================================================"

touch logs/SWIMMER_ALL_8_SUCCESS
