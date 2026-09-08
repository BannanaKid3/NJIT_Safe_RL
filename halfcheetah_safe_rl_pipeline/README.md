# HalfCheetah Safe-RL Method Pipeline

This folder contains the reproducible source-code workflow used to design, train, evaluate, and test the HalfCheetah state constraints.

## Workflow

1. `01_cost_design`: train an unconstrained baseline, collect raw torso-pitch and six-joint-speed measurements, and analyze candidate thresholds.
2. `02_cost_definitions`: define the pitch, joint-speed, scalar-combined, multi-constraint, and gravity-perturbed environments.
3. `03_nominal_training`: train pitch-only, joint-speed-only, scalar-combined, and multi-constraint policies.
4. `04_nominal_evaluation`: evaluate selected checkpoints and compare reward, feasibility, individual constraints, and combined costs.
5. `05_action_constraint_comparison`: compare the original action-constrained policy with the state-constrained policies.
6. `06_gravity_robustness`: train under per-step gravity noise and evaluate at multiple gravity-disturbance levels.
7. `setup`: environment and dependency specifications.

## Scalar combined cost

The scalar policy is trained using:

C_combined = max(C_pitch, C_joint)

Pitch and joint-speed components should also be recorded separately during training for diagnostics and plotting. They must not replace the scalar combined cost used by the policy.

Generated datasets, model checkpoints, logs, CSV files, plots, GIFs, ZIP files, and notebook outputs are intentionally excluded.
