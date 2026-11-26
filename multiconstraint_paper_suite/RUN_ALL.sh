#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-.}"
bash "$(dirname "$0")/training_plotting/scripts/00_inventory.sh" "$REPO"
bash "$(dirname "$0")/training_plotting/scripts/01_ablation.sh" "$REPO"
bash "$(dirname "$0")/training_plotting/scripts/02_robustness.sh" "$REPO"
bash "$(dirname "$0")/evaluation/scripts/04_all_evaluation.sh" "$REPO"
