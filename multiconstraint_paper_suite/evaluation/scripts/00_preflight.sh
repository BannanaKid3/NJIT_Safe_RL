#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-.}"
python "$(dirname "$0")/../src/preflight.py" --repo "$REPO"
