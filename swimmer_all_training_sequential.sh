#!/usr/bin/env bash
set -euo pipefail

cd /home/bannanakid/RCRL_on_RCMDP
mkdir -p logs

python iclr2027_safe_rl_extension_pack_v2/tools/preflight.py \
  --repo . \
  --require-swimmer

python -m py_compile \
  code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py \
  code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
  code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
  code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py

echo "=== 1/8 Swimmer Multi NP seed 2 run9612 ==="
python code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py \
  --run 9612 --seed 2 \
  --persistent_eps 0.1 \
  --beta 30000 \
  --K_epochs 5 \
  --max_train_steps 16000 \
  --warm_start_episode 150 \
  --lr_cost 1e-3 \
  --entropy_coef 0.001 \
  --weight_reg 0.0 \
  --sigma_viscosity 0.0 \
  --cost_dim 2 \
  --cost_scale 1.0 \
  2>&1 | tee logs/swimmer_multi_np_seed2_run9612.log \
&& \
echo "=== 2/8 Swimmer Combined NP seed 2 run9712 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
  --run 9712 --seed 2 \
  --persistent_eps 0.1 \
  --beta 30000 \
  --K_epochs 5 \
  --max_train_steps 16000 \
  --warm_start_episode 150 \
  --lr_cost 1e-3 \
  --entropy_coef 0.001 \
  --weight_reg 0.0 \
  --sigma_viscosity 0.0 \
  2>&1 | tee logs/swimmer_combined_np_seed2_run9712.log \
&& \
echo "=== 3/8 Swimmer Multi NP seed 5 run9615 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py \
  --run 9615 --seed 5 \
  --persistent_eps 0.1 \
  --beta 30000 \
  --K_epochs 5 \
  --max_train_steps 16000 \
  --warm_start_episode 150 \
  --lr_cost 1e-3 \
  --entropy_coef 0.001 \
  --weight_reg 0.0 \
  --sigma_viscosity 0.0 \
  --cost_dim 2 \
  --cost_scale 1.0 \
  2>&1 | tee logs/swimmer_multi_np_seed5_run9615.log \
&& \
echo "=== 4/8 Swimmer Combined NP seed 5 run9715 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_swimmer_combined.py \
  --run 9715 --seed 5 \
  --persistent_eps 0.1 \
  --beta 30000 \
  --K_epochs 5 \
  --max_train_steps 16000 \
  --warm_start_episode 150 \
  --lr_cost 1e-3 \
  --entropy_coef 0.001 \
  --weight_reg 0.0 \
  --sigma_viscosity 0.0 \
  2>&1 | tee logs/swimmer_combined_np_seed5_run9715.log \
&& \
echo "=== 5/8 Swimmer Multi P+R seed 2 run9812 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
  --run 9812 --seed 2 \
  --persistent_eps 0.1 \
  --beta 30000 \
  --K_epochs 5 \
  --max_train_steps 16000 \
  --warm_start_episode 150 \
  --lr_cost 1e-3 \
  --entropy_coef 0.001 \
  --weight_reg 0.001 \
  --sigma_viscosity 0.7 \
  --cost_dim 2 \
  --cost_scale 1.0 \
  2>&1 | tee logs/swimmer_multi_pr_seed2_run9812.log \
&& \
echo "=== 6/8 Swimmer Combined P+R seed 2 run9912 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py \
  --run 9912 --seed 2 \
  --persistent_eps 0.1 \
  --beta 30000 \
  --K_epochs 5 \
  --max_train_steps 16000 \
  --warm_start_episode 150 \
  --lr_cost 1e-3 \
  --entropy_coef 0.001 \
  --weight_reg 0.001 \
  --sigma_viscosity 0.7 \
  2>&1 | tee logs/swimmer_combined_pr_seed2_run9912.log \
&& \
echo "=== 7/8 Swimmer Multi P+R seed 5 run9815 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py \
  --run 9815 --seed 5 \
  --persistent_eps 0.1 \
  --beta 30000 \
  --K_epochs 5 \
  --max_train_steps 16000 \
  --warm_start_episode 150 \
  --lr_cost 1e-3 \
  --entropy_coef 0.001 \
  --weight_reg 0.001 \
  --sigma_viscosity 0.7 \
  --cost_dim 2 \
  --cost_scale 1.0 \
  2>&1 | tee logs/swimmer_multi_pr_seed5_run9815.log \
&& \
echo "=== 8/8 Swimmer Combined P+R seed 5 run9915 ===" \
&& \
python code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py \
  --run 9915 --seed 5 \
  --persistent_eps 0.1 \
  --beta 30000 \
  --K_epochs 5 \
  --max_train_steps 16000 \
  --warm_start_episode 150 \
  --lr_cost 1e-3 \
  --entropy_coef 0.001 \
  --weight_reg 0.001 \
  --sigma_viscosity 0.7 \
  2>&1 | tee logs/swimmer_combined_pr_seed5_run9915.log

echo
echo "ALL EIGHT SWIMMER TRAINING RUNS FINISHED SUCCESSFULLY."
