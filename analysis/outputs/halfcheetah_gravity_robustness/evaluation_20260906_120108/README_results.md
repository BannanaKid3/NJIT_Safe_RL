# HalfCheetah Gravity-Robustness Evaluation

Each selected policy was evaluated deterministically for 100 episodes per gravity level on reset seeds 2301–2400.
Gravity was independently redrawn before every MuJoCo step and therefore affected the next state, reward, and costs.

## Method means across training seeds

| Method | Train σ | Eval σ | Reward | State feasible | Mean max cost |
|---|---:|---:|---:|---:|---:|
| Multi-constraint | 0.00 | 0.00 | 2327.3 | 85.0% | 0.0835 |
| Multi-constraint | 0.00 | 1.00 | 2221.3 | 81.3% | 0.0997 |
| Multi-constraint | 0.00 | 2.00 | 2083.0 | 80.2% | 0.0887 |
| Multi-constraint | 0.50 | 0.00 | 3354.9 | 78.5% | 0.0960 |
| Multi-constraint | 0.50 | 1.00 | 3283.5 | 82.2% | 0.0792 |
| Multi-constraint | 0.50 | 2.00 | 3100.3 | 72.2% | 0.1046 |
| Scalar combined | 0.00 | 0.00 | 3459.2 | 91.0% | 0.0537 |
| Scalar combined | 0.00 | 1.00 | 3405.4 | 88.2% | 0.0642 |
| Scalar combined | 0.00 | 2.00 | 3146.6 | 81.2% | 0.1212 |
| Scalar combined | 0.50 | 0.00 | 2121.2 | 92.3% | 0.0470 |
| Scalar combined | 0.50 | 1.00 | 2090.4 | 91.8% | 0.0550 |
| Scalar combined | 0.50 | 2.00 | 1979.3 | 84.8% | 0.0842 |

## Interpretation rules

- Robustness is shown as degradation as evaluation σ increases, not performance at a single perturbed setting.
- The nominal policies are required baselines for deciding whether perturbed training helped.
- The 100 reset episodes characterize each fixed policy. Algorithm-level uncertainty comes from independent training seeds.
- There are 4 training seeds per condition, so seed-level conclusions remain limited.
- Exact sign-flip p-values and paired seed differences are in `seed_level_paired_training_comparisons.csv`.
- `rho` is kept separate from gravity σ: σ controls environment variation; rho controls the existing IPM critic-norm term.

## Primary outputs

- `episode_metrics.csv`: one row per policy, evaluation σ, and reset episode.
- `policy_summary.csv`: 100-episode summary for every selected policy.
- `method_summary_across_training_seeds.csv`: method means, SDs, and 95% t intervals across training seeds.
- `robustness_retention.csv`: reward retention and feasibility change relative to σ=0.
- `seed_level_paired_method_comparisons.csv`: scalar-versus-multi comparisons across matched training seeds.
- `within_seed_matched_reset_tests.csv`: fixed-policy paired-reset diagnostics, kept separate from seed-level inference.
- `selected_episode_timestep_metrics.csv`: raw gravity, pitch, joint speed, action, reward, and cost traces.

No paired training-seed comparison reached p < 0.05 under the exact sign-flip test.
