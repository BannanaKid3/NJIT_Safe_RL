# Multi-Constraint Paper Figure and Evaluation Suite

Use this from the repository root:

```bash
cd /home/bannanakid/RCRL_on_RCMDP
unzip -o multiconstraint_paper_suite.zip -d .
chmod +x multiconstraint_paper_suite/evaluation/scripts/*.sh
chmod +x multiconstraint_paper_suite/training_plotting/scripts/*.sh
```

## 1. Verify checkpoints

```bash
bash multiconstraint_paper_suite/evaluation/scripts/00_preflight.sh .
```

Do not continue if a required paper checkpoint is missing.

## 2. Priority requested by Shilpa: Combined Cost vs Multi Constraint

```bash
bash multiconstraint_paper_suite/evaluation/scripts/01_priority_ablation.sh .
```

This runs deterministic 100-episode inference for the completed Multi and Combined policies and creates the primary reward and max-cost ablation figures for Quadrotor, HalfCheetah, and Swimmer. Where P+R checkpoints exist, it creates the perturbed ablation too.

Primary evaluation settings:

- Quadrotor NP ablation: nominal environment.
- Quadrotor P+R ablation: uniform dynamics disturbance ±0.005.
- HalfCheetah NP ablation: gravity sigma 0.
- HalfCheetah P+R ablation: gravity sigma 1.5.
- Swimmer NP ablation: viscosity sigma 0.
- Swimmer P+R ablation: viscosity sigma 1.0.
- 100 evaluation episodes per training seed.
- Shared reset seeds 2301 through 2400.
- Deterministic mean action.
- Main safety statistic is the maximum over the component peak costs.

Outputs go to:

```text
paper_evaluation/
  quadrotor/
  halfcheetah/
  swimmer/
```

Each condition includes raw arrays, `seed_summary.csv`, `method_summary.csv`, Shilpa-style reward/max-cost figures, separate legends, and component diagnostics.

## 3. Multi-constraint adaptation story

```bash
bash multiconstraint_paper_suite/evaluation/scripts/02_adaptation.sh .
```

This creates the requested Surrogate Objective NP vs Ours P+R inference comparisons:

- Quadrotor: evaluated under uniform ±0.005 dynamics disturbance.
- HalfCheetah: evaluated at gravity sigma 1.5.
- Swimmer: evaluated at viscosity sigma 1.0.

The plot labels and colors follow Shilpa's current figures: `Ours` in blue and `Surrogate Obj` in orange.

## 4. Quadrotor robustness results

```bash
bash multiconstraint_paper_suite/evaluation/scripts/03_quadrotor_robustness.sh .
```

This evaluates the requested multi-constraint robustness comparisons under:

```text
uniform ±0.005
uniform ±0.01
uniform ±0.05
white noise 0.01
white noise 0.05
horizontal step 0.005
horizontal step 0.01
```

For each disturbance it creates both the NP vs P+R comparison and the P+R Multi vs Combined ablation.

## 5. Run everything

```bash
bash multiconstraint_paper_suite/evaluation/scripts/04_all_evaluation.sh .
```

The Quadrotor robustness sweep is substantially larger than the priority ablation. Use steps 2 and 3 first if the immediate goal is to send Shilpa the requested ablation figures.

## 6. Training figures

The prior training plotting package is included under `training_plotting/`.

```bash
bash multiconstraint_paper_suite/training_plotting/scripts/00_inventory.sh .
bash multiconstraint_paper_suite/training_plotting/scripts/01_ablation.sh .
bash multiconstraint_paper_suite/training_plotting/scripts/02_robustness.sh .
```

Current Shilpa-style training figures use full per-episode histories, an 80-episode moving average, and mean ± 1 SD across runs.

## 7. Inference figure formatting

The primary inference figures follow the current HalfCheetah/Swimmer comparison scripts in Shilpa's repository:

- 20-episode simple moving average.
- 44 × 28 reward figure.
- 42 × 28 max-cost figure.
- font size 100.
- line width 15.
- label font size 130.
- no grid.
- no legend inside the main figure.
- separate horizontal and vertical legend images.
- dashed black safety threshold.
- blue safe-region shading and red unsafe-region shading.

The primary inference plots average the matched training-seed policies at each evaluation episode before smoothing. Seed-level raw results and summaries are saved separately.

## 8. Package results

```bash
bash multiconstraint_paper_suite/evaluation/scripts/05_pack_results.sh .
```

This creates:

```text
paper_evaluation_results.zip
```
