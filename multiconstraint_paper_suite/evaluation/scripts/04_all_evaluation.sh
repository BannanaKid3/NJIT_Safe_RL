#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-.}"
"$(dirname "$0")/00_preflight.sh" "$REPO"
"$(dirname "$0")/01_priority_ablation.sh" "$REPO"
"$(dirname "$0")/02_adaptation.sh" "$REPO"
"$(dirname "$0")/03_quadrotor_robustness.sh" "$REPO"
