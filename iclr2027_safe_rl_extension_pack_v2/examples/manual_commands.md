# Manual command equivalents

Run these from the root of `RCRL_on_RCMDP` after running `setup_pipeline.py`. The PowerShell launchers in `scripts/` are preferred because they consistently use seeds 2 and 5 and keep logs.

## Quadrotor Combined, nominal

```powershell
python .\code_ipm_rcmdp_rcrl_max_quadrotor_combined.py --run 201 --seed 2 --persistent_eps 0.5 --beta 30000 --K_epochs 5 --max_train_steps 40000 --warm_start_episode 1000 --lr_cost 5e-4 --lr_c 5e-4 --lr_a 5e-4 --entropy_coef 0.007 --weight_reg 0.0 --batch_size 1024 --mini_batch_size 256 --cost_scale 100.0
```

Repeat with `--run 205 --seed 5`.

## Quadrotor Multi, perturbed + robust

```powershell
python .\code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py --run 301 --seed 2 --persistent_eps 0.5 --beta 30000 --K_epochs 5 --max_train_steps 40000 --warm_start_episode 1000 --lr_cost 5e-4 --lr_c 5e-4 --lr_a 5e-4 --entropy_coef 0.007 --weight_reg 0.001 --batch_size 1024 --mini_batch_size 256 --cost_scale 100.0 --cost_dim 4
```

Repeat with seed 5/run305.

## Swimmer calibration

```powershell
python .\iclr2027_safe_rl_extension_pack_v2\calibration\calibrate_swimmer_joint_speed.py --repo . --model-prefix .\models\SwimmerWithPos\run2\Best_RCAC --episodes 30
```

The model prefix is the path before `_actor`. If only one nominal `Best_RCAC_actor` exists, omit `--model-prefix` and the calibrator will auto-select it.

## Swimmer Multi, nominal

```powershell
python .\code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py --run 501 --seed 2 --env SwimmerWithPos --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.0 --sigma_viscosity 0.0 --cost_dim 2 --cost_scale 1.0
```

## Swimmer Combined, nominal

```powershell
python .\code_ipm_rcmdp_rcrl_max_swimmer_combined.py --run 601 --seed 2 --env SwimmerWithPos --persistent_eps 0.1 --beta 30000 --K_epochs 5 --max_train_steps 16000 --warm_start_episode 150 --lr_cost 1e-3 --entropy_coef 0.001 --weight_reg 0.0 --sigma_viscosity 0.0
```
