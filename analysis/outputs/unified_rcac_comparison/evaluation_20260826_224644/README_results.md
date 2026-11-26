# Unified RCAC HalfCheetah Evaluation

All policies were evaluated deterministically on the same reset seeds. Every policy was measured using Shilpa's action constraint, the pitch constraint, and all six joint-speed constraints regardless of its training objective.

## Method means across training seeds

| Method | Reward | Action feasible | State feasible | All feasible |
|---|---:|---:|---:|---:|
| Shilpa action | 2138.4 | 87.8% | 99.8% | 87.5% |
| Scalar combined | 3476.4 | 0.0% | 91.2% | 0.0% |
| Multi-constraint | 2320.9 | 0.0% | 81.7% | 0.0% |

## Interpretation rules

- Reward and feasibility are separate outcomes; a method is not universally better unless it is no worse on both.
- Raw action-cost, pitch-cost, and joint-cost magnitudes are not directly compared because their scaling differs.
- Algorithm-level claims use independent training seeds. The 100 matched resets characterize each selected policy but are not 100 independent training runs.
- With 4 training seeds, the exact two-sided sign-flip test cannot reach p < 0.05 when only four nonzero paired differences are available; confidence intervals and seed consistency are therefore emphasized.
- `pareto_by_seed.csv` identifies methods that are not dominated simultaneously on reward and safety.

## Shilpa OG cost

Shilpa's original cost is `max(max(abs(action)) - 0.5, 0)`. It controls peak action magnitude, an upstream control-effort proxy. It does not directly measure torso pitch or joint speed, but limiting actions can indirectly reduce both.

## Primary evidence files

- `episode_metrics.csv`: one row per policy and evaluation episode.
- `policy_summary.csv`: one row per selected trained policy.
- `method_summary_across_training_seeds.csv`: algorithm means across independent seeds.
- `seed_level_pairwise_tests.csv`: paired training-seed comparisons.
- `within_seed_paired_reset_tests.csv`: matched-reset comparisons for fixed selected policies.
- `pareto_by_seed.csv`: reward-safety dominance results.
- `selected_episode_timestep_metrics.csv`: raw traces used in matched time-series figures.

## Automatic statistical caution

No seed-level pairwise comparison reached p < 0.05 under the exact paired sign-flip test. Treat method advantages as descriptive unless more independent training seeds are added.

## Pareto summary

- reward_vs_all_constraint_feasibility: Multi-constraint was nondominated in 1/4 seeds.
- reward_vs_all_constraint_feasibility: Scalar combined was nondominated in 2/4 seeds.
- reward_vs_all_constraint_feasibility: Shilpa action was nondominated in 4/4 seeds.
- reward_vs_state_feasibility: Multi-constraint was nondominated in 2/4 seeds.
- reward_vs_state_feasibility: Scalar combined was nondominated in 3/4 seeds.
- reward_vs_state_feasibility: Shilpa action was nondominated in 4/4 seeds.
