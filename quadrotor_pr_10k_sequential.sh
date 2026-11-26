#!/usr/bin/env bash
set -euo pipefail

cd /home/bannanakid/RCRL_on_RCMDP
mkdir -p logs

echo "=== Quadrotor P+R 10k paired experiment ==="

# Confirm the intended perturbation profile before spending compute.
PROFILE=$(cat envs/env_configs/ACTIVE_PERTURBATION_PROFILE.txt)
if [ "$PROFILE" != "uniform_0p005" ]; then
    echo "ERROR: Expected uniform_0p005 but got: $PROFILE"
    exit 1
fi

python -m py_compile \
  code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py \
  code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py

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
  --weight_reg 0.001
  --batch_size 1024
  --mini_batch_size 256
  --cost_scale 100.0
)

echo "=== 1/4 Quad Multi P+R seed 2 run9412 ==="
python code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py \
  --run 9412 --seed 2 "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/quad_multi_pr_seed2_run9412.log \
&& \
echo "=== 2/4 Quad Combined P+R seed 2 run9512 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py \
  --run 9512 --seed 2 "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/quad_combined_pr_seed2_run9512.log \
&& \
echo "=== 3/4 Quad Multi P+R seed 5 run9415 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py \
  --run 9415 --seed 5 "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/quad_multi_pr_seed5_run9415.log \
&& \
echo "=== 4/4 Quad Combined P+R seed 5 run9515 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py \
  --run 9515 --seed 5 "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/quad_combined_pr_seed5_run9515.log

echo
echo "ALL FOUR QUADROTOR P+R RUNS FINISHED SUCCESSFULLY."
