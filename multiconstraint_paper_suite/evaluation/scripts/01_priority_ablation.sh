#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-.}"
SRC="$(cd "$(dirname "$0")/../src" && pwd)"
python "$SRC/eval_quadrotor.py" --repo "$REPO" --task np_ablation
python "$SRC/eval_quadrotor.py" --repo "$REPO" --task pr_ablation
python "$SRC/eval_halfcheetah.py" --repo "$REPO" --task np_ablation
python "$SRC/eval_halfcheetah.py" --repo "$REPO" --task pr_ablation
python "$SRC/eval_swimmer.py" --repo "$REPO" --task np_ablation
python "$SRC/eval_swimmer.py" --repo "$REPO" --task pr_ablation
