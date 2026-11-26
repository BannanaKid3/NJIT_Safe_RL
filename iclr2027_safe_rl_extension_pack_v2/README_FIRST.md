# ICLR 2027 Safe-RL Extension Pack v2

This is a **drop-in overlay** for Shilpa's current `RCRL_on_RCMDP` repository. It was source-matched against the public `master` snapshot inspected on 2026-09-19, latest visible commit `fc741df13065458077e24f7b07227d6a12c2048b`.

## What is fixed in v2

- Quadrotor Combined now uses the current Multi formulation correctly: four residual costs are preserved and the scalar ablation is `C_combined = max(C0,C1,C2,C3)`.
- Quadrotor perturbed + robust trainers load **dedicated active disturbance YAMLs**, not the nominal files with commented disturbance examples.
- The P+R trainers abort if the instantiated environment has no active disturbances.
- Eight explicit Quadrotor disturbance profiles are included. The default is reproducible but provisional until Shilpa confirms the exact unpublished YAML/profile used for the final experiment.
- Swimmer now has a complete two-cost implementation: the existing actuator/action exceedance cost is preserved as C1, a calibrated internal-joint-speed exceedance is added as C2, and the scalar ablation uses `max(C1,C2)` on those exact same components.
- Swimmer training is intentionally blocked until the joint-speed calibration JSON is populated from a nominal trained Swimmer policy.
- Two-seed launchers consistently use seeds 2 and 5.
- Analysis follows the safety-first checkpoint-selection rule and plotting uses mean across the two seeds, ±1.5 SD, window-10 smoothing with the first 10 points untouched, actual costs rather than safety percentages, worst/MAX cost, and individual vector costs when available.

## Why the ZIP generates the full trainer files instead of shipping copied 2,000-line files

`setup_pipeline.py` patches the exact current trainer files already in your clone and writes **new full trainer files** beside them. It does not overwrite Shilpa's originals. This keeps the experiment code tied to the updated repo snapshot and makes an upstream mismatch fail loudly rather than silently running stale code.

The patch anchors were checked against the current `master`: the Quad scalar source has exactly two active weighted-cost blocks, the Quad Multi train/eval blocks each match once, the robust YAML anchors each match once, and the current Swimmer import anchor matches once.

## Start here

Place the extracted folder directly inside your cloned repository:

```text
RCRL_on_RCMDP/
├── code_ipm_....py
├── envs/
├── ...
└── iclr2027_safe_rl_extension_pack_v2/
```

Then, from the **repo root**:

```powershell
powershell -ExecutionPolicy Bypass -File .\iclr2027_safe_rl_extension_pack_v2\scripts\00_install_and_preflight.ps1 -Repo . -QuadProfile white_noise_0p01
```

After that, follow `RUN_ORDER.md` in numerical order.

## Important Quadrotor YAML note

The committed repo does not contain a separate active perturbed YAML pair. The profile files here make perturbation explicit and reproducible. `white_noise_0p01` is the convenience default because that magnitude appears in the recorded robustness sweep, **not** a claim that it is Shilpa's exact unpublished final-paper configuration. If Shilpa sends the exact YAMLs/settings, either add them as a profile or copy them to:

```text
envs/env_configs/constrained_tracking_train_perturbed.yaml
envs/env_configs/constrained_tracking_eval_perturbed.yaml
```

and leave the runtime disturbance guard in place.

## Validation status

The package's Python files compile, the local cost/smoothing unit checks pass, and all patch anchors were source-matched against Shilpa's current `master`. Long MuJoCo/PyBullet training itself was **not executed here**, because that requires the repository's simulator environment and compute on your machine. Run `tools/preflight.py` after installation before launching long jobs.
