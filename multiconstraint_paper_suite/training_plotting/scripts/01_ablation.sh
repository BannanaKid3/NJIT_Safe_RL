#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
python "$ROOT/multiconstraint_plot_suite/src/make_all.py" --repo "$ROOT" --only ablation
