# Exact file placement

You do **not** manually scatter the package files around the repository. Put the whole folder `iclr2027_safe_rl_extension_pack_v2/` at the repository root, then run `setup_pipeline.py` or `scripts/00_install_and_preflight.ps1`.

The installer creates/copies the following files in `RCRL_on_RCMDP`:

| Destination | What it does |
|---|---|
| `experiment_extensions/quadrotor_costs.py` | extracts all 4 Quad residuals and computes `max(C0..C3)` |
| `experiment_extensions/quadrotor_disturbance_guard.py` | refuses a P/P+R run if `env.disturbances` is empty |
| `experiment_extensions/swimmer_costs.py` | shared C1/C2 definitions |
| `experiment_extensions/swimmer_multi_constraint_env.py` | nominal/perturbed Multi and Combined Swimmer wrappers |
| `experiment_extensions/swimmer_thresholds.json` | calibration values; initially blocked as `NOT_CALIBRATED` |
| `envs/env_configs/constrained_tracking_train_perturbed.yaml` | selected active Quad training perturbation |
| `envs/env_configs/constrained_tracking_eval_perturbed.yaml` | selected active Quad evaluation perturbation |
| `envs/env_configs/ACTIVE_PERTURBATION_PROFILE.txt` | records which packaged profile was selected |
| `code_ipm_rcmdp_rcrl_max_quadrotor_combined.py` | full nominal scalar Combined trainer generated from current Quad scalar source |
| `code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py` | full Quad Combined P+R trainer |
| `code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py` | full Quad Multi P+R trainer |
| `code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint.py` | full nominal two-cost Swimmer Multi trainer |
| `code_ipm_rcmdp_rcrl_max_swimmer_multi_constraint_perturbed.py` | full Swimmer Multi P+R trainer |
| `code_ipm_rcmdp_rcrl_max_swimmer_combined.py` | full nominal Swimmer Combined trainer |
| `code_ipm_rcmdp_rcrl_max_swimmer_combined_perturbed.py` | full Swimmer Combined P+R trainer |

The original Shilpa trainer files are left untouched. Re-running setup also preserves a `CALIBRATED` Swimmer threshold JSON instead of overwriting it.

## What stays inside the package folder

The `scripts/`, `analysis/`, `calibration/`, `paper_plotting/`, `tools/`, `tests/`, and `env_profiles/` directories remain inside `iclr2027_safe_rl_extension_pack_v2/`. The launchers call them from there.
