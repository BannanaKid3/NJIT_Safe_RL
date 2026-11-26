# Validation report

Build date: 2026-09-19

## Source matching against current Shilpa `master`

The package patch anchors were checked against the current public repository snapshot before packaging:

- Quad single active weighted-cost blocks found: **2** (expected 2: train + eval)
- Quad Multi exact evaluation block found: **1**
- Quad Multi exact training block found: **1**
- Quad robust train YAML anchor found: **1**
- Quad robust eval YAML anchor found: **1**
- Quad robust environment-reset anchor found: **1**
- Swimmer scalar import anchor found: **1**

The source snapshot identifiers are in `SOURCE_SNAPSHOT.md`.

## Local static/unit checks

- All package Python files successfully passed `py_compile`.
- Quadrotor cost helper checked for clipping/scaling and `max(C0..C3)` behavior.
- Swimmer C1/C2 helper checked on deterministic sample inputs.
- Smoothing helper checked to preserve indices 0-9 and begin trailing-window-10 smoothing at index 10.
- Two-seed checkpoint-analysis script tested with synthetic scalar/vector-cost runs.
- Paper plotting script tested with synthetic two-seed runs and produced reward, worst-cost, and component-cost figures.
- Quad Combined replacement indentation was checked separately for both evaluation-level and nested training-level indentation.
- All 16 packaged Quadrotor profile YAMLs parsed successfully and contain an active `disturbances.dynamics` block.

## Not run in this environment

Long MuJoCo or PyBullet training was not executed here. That requires the simulator/dependency environment from your clone and your local compute. `tools/preflight.py` is included so the generated full trainers are syntax-checked after installation, before you spend time on long runs.
