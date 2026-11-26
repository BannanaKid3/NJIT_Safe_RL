QUADROTOR SPEED PATCH

Files patched in-place:
- code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint.py
- code_ipm_rcmdp_rcrl_max_quadrotor_combined.py
- code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_perturbed.py
- code_ipm_rcmdp_rcrl_max_quadrotor_combined_robust.py
- code_ipm_rcmdp_rcrl_max_quadrotor_multi_constraint_robust.py (if present)

Changes:
- Active env.render() calls are disabled.
- The expensive per-episode training-history save/plot block runs every
  100 episodes and on the final episode instead of every episode.

Not changed:
- RL algorithm or actor/critic update logic
- costs, constraints, reward
- physics
- disturbances
- seeds/hyperparameters
- evaluate_freq
- best-checkpoint selection and saving

The patcher makes .pre_io_opt.bak backups and py_compile-validates each file.

Typical use:
  cd /home/bannanakid/RCRL_on_RCMDP
  python /path/to/optimize_quadrotor_io.py --repo .
  bash /path/to/run_quadrotor_np_10k_sequential.sh
