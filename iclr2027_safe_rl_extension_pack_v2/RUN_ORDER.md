# Exact experiment order

This package is arranged around the current paper plan, not around generic RL experimentation.

## 0. Install once and freeze the Quadrotor perturbation profile

From the root of your cloned `RCRL_on_RCMDP` repository, with this package folder sitting inside it:

```powershell
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\00_install_and_preflight.ps1 -Repo . -QuadProfile white_noise_0p01
```

For final paper-facing runs, replace `white_noise_0p01` with the exact profile/settings Shilpa confirms. The included profiles are reproducible options recovered from the committed YAML comments and meeting notes; `white_noise_0p01` is only the convenience default.

The generated P+R Quad trainers contain a runtime guard. If `env.disturbances` is `None` or empty, they stop instead of silently producing a nominal run with a perturbed label.

## 1. Quadrotor Combined, nominal, two seeds

```powershell
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\01_quad_combined_two_seeds.ps1 -Repo .
```

This uses seeds 2 and 5 and writes to independent namespaces:

- `data_train/QuadrotorCombined/run201`
- `data_train/QuadrotorCombined/run205`
- `models/QuadrotorCombined/run201`
- `models/QuadrotorCombined/run205`

The scalar safety cost is exactly `max(C0,C1,C2,C3)` after the same clipping/scaling used by the current 4-D Multi trainer.

## 2. Analyze Quadrotor Combined

```powershell
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\02_analyze_quad_combined.ps1 -Repo .
```

If you know the two existing Multi NP data folders, pass them too:

```powershell
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\02_analyze_quad_combined.ps1 `
  -Repo . `
  -ExistingMultiRunSeed2 ".\data_train\Quadrotor\run<seed2-run>" `
  -ExistingMultiRunSeed5 ".\data_train\Quadrotor\run<seed5-run>"
```

The selector uses a safety-first rule across the two seeds: highest mean reward among checkpoints feasible in both seeds; if none are jointly feasible, lowest mean worst cost with reward as tie-break.

## 3. Calibrate the second Swimmer constraint

Do this before running any generated Swimmer trainer:

```powershell
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\05_swimmer_calibrate.ps1 -Repo .
```

If multiple nominal Swimmer `Best_RCAC_actor` checkpoints exist, pass a specific model prefix, for example:

```powershell
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\05_swimmer_calibrate.ps1 `
  -Repo . -ModelPrefix ".\models\SwimmerWithPos\run2\Best_RCAC" -Episodes 30
```

The calibration writes `experiment_extensions/swimmer_thresholds.json`. Generated Swimmer environments refuse to initialize while that file remains `NOT_CALIBRATED`.

## 4. Swimmer nominal Multi and Combined, two seeds each

```powershell
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\06_swimmer_multi_two_seeds.ps1 -Repo .
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\07_swimmer_combined_two_seeds.ps1 -Repo .
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\08_analyze_swimmer.ps1 -Repo .
```

Swimmer uses the same two components for both methods:

`C1 = actuator-action exceedance above 0.5`, preserving the existing repo Swimmer cost magnitude

`C2 = calibrated internal-joint-speed exceedance`; the default calibration maps the 99.5th-percentile nominal joint-speed sample to cost 0.1 after setting the zero-cost threshold at the 95th percentile

Multi keeps `[C1,C2]`; Combined uses `max(C1,C2)`.

## 5. Perturbed + robust runs

Once the nominal ablation behaves sensibly, the included launchers are:

```powershell
# Quadrotor Multi P+R using the selected explicit YAML profile
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\03_quad_multi_pr_two_seeds.ps1 -Repo .

# Optional Quadrotor Combined P+R
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\04_quad_combined_pr_optional_two_seeds.ps1 -Repo .

# Swimmer Multi P+R, repo-consistent default sigma_viscosity=0.7
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\09_swimmer_multi_pr_two_seeds.ps1 -Repo .

# Optional Swimmer Combined P+R
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\10_swimmer_combined_pr_optional_two_seeds.ps1 -Repo .
```

The optional Combined P+R runs are there for the robustness ablation. You do not need to spend compute on them before the nominal comparison is checked.
