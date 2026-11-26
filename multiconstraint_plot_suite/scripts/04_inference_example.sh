#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
python "$ROOT/multiconstraint_plot_suite/src/plot_inference.py" \
  --method "Multi Constraint" "$ROOT/inference_data/multi.npz" \
  --method "Combined Cost" "$ROOT/inference_data/combined.npz" \
  --threshold 0.1 \
  --outdir "$ROOT/multiconstraint_figures/inference_example" \
  --prefix inference
