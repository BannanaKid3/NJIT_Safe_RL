#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-.}"
cd "$REPO"
rm -f paper_evaluation_results.zip
zip -qr paper_evaluation_results.zip paper_evaluation
printf '%s\n' "$PWD/paper_evaluation_results.zip"
