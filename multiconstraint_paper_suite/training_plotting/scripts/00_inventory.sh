#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-.}"
python "$ROOT/multiconstraint_plot_suite/src/inventory.py" --repo "$ROOT"
