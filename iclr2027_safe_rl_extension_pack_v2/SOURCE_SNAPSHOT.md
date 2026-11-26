# Source snapshot used for v2

Repository: `shilpa2301/RCRL_on_RCMDP`

Visible `master` snapshot inspected 2026-09-19:

- latest visible commit: `fc741df13065458077e24f7b07227d6a12c2048b`
- Quad single trainer blob: `585124a4968406f3ab96858ac0df58ad9a6c7cd3`
- Quad multi trainer blob: `c8a2e62297505a2d760fedbf4ac69504d105d903`
- Quad robust multi trainer blob: `b0d6d6d50aa2d64286c9e7ae93480464b4bf2488`
- Swimmer scalar trainer blob: `0cdd3ffd3ba41afe5faace787c5447df43057a54`
- `envs/swimmer.py` blob: `7f7de1b9b22315f94dd403498f4de467f7c624e2`

Observed facts used by this package:

- Current Quadrotor Multi uses `cost_dim=4`.
- The 2-D input bound has lower `[0.0,0.0]` and upper `[0.12,0.12]`, producing four residuals.
- The current committed reset/eval YAML disturbance examples are commented out.
- The current robust trainer still points to the nominal reset/eval YAML names.
- Current Swimmer uses `ACTION_TORQUE_THRESHOLD = 0.5` and its perturbed class applies per-step Gaussian viscosity perturbation controlled by `sigma_viscosity`.
