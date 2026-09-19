#!/usr/bin/env bash
# Validate a submission CSV and submit it to the competition (5 submissions/day!).
#
#   scripts/submit.sh outputs/classical-baseline/submission.csv "classical baseline, ratio=0.85"
#   scripts/submit.sh --check outputs/classical-baseline/submission.csv     # validate only
set -euo pipefail
unset VIRTUAL_ENV
cd "$(dirname "$0")/.."
competition="filament-segmentation-2026"

check_only=""
if [[ "${1:-}" == "--check" ]]; then check_only=1; shift; fi
csv="${1:?usage: scripts/submit.sh [--check] <submission.csv> <message>}"

uv run python scripts/check_submission.py "$csv"
if [[ -n "$check_only" ]]; then exit 0; fi

message="${2:?a submission message is required}"
uv run kaggle competitions submission-limits "$competition"
read -r -p "Submit $csv to $competition? [y/N] " answer
[[ "$answer" == "y" || "$answer" == "Y" ]] || { echo "aborted"; exit 1; }

uv run kaggle competitions submit "$competition" -f "$csv" -m "$message"
sleep 20
uv run kaggle competitions submissions "$competition" | head -8
