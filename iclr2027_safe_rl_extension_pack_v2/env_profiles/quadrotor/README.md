# Quadrotor perturbation profiles

Profiles included: white-noise 0.005/0.01/0.05, uniform ±0.005/±0.01/±0.05, and horizontal step 0.005/0.01.

The installer copies the selected pair to `envs/env_configs/constrained_tracking_train_perturbed.yaml` and `...eval_perturbed.yaml`.

`white_noise_0p01` is a convenience default because it appears in the recorded robustness sweep, **not** a claim that it is Shilpa's exact unpublished paper YAML. If she gives you the exact local YAML/profile, use that exact setting for final paper-facing runs.
