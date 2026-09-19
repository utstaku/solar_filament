#!/usr/bin/env bash
# Build the notebook for a kernel folder, push it to Kaggle, wait for the run, fetch log and small outputs.
#
#   scripts/kernel_run.sh kernels/classical-baseline            # CPU (or whatever kernel-metadata.json says)
#   scripts/kernel_run.sh kernels/train-unet --accelerator NvidiaTeslaT4
#   OUTPUT_PATTERN='.*' scripts/kernel_run.sh kernels/foo       # download every output file
#
# Only files matching OUTPUT_PATTERN are downloaded, so model weights and images stay on Kaggle by
# default; figures are rendered inside the notebook, which is viewable at the printed URL.
set -euo pipefail
unset VIRTUAL_ENV
cd "$(dirname "$0")/.."

kernel_dir="${1:?usage: scripts/kernel_run.sh <kernel-dir> [extra 'kaggle kernels push' args]}"
shift
kernel_dir="${kernel_dir%/}"
pattern="${OUTPUT_PATTERN:-.*\.(csv|json|txt)$}"
poll_sec="${POLL_SEC:-30}"

build_dir="$(uv run python scripts/build_notebook.py "$kernel_dir")"
kernel_id="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$build_dir/kernel-metadata.json")"
out_dir="outputs/$(basename "$kernel_dir")"

echo ">> built $build_dir"
echo ">> pushing as $kernel_id"
uv run kaggle kernels push -p "$build_dir" "$@"
echo ">> https://www.kaggle.com/code/$kernel_id"

while true; do
  status="$(uv run kaggle kernels status "$kernel_id" 2>&1 | tail -1)"
  echo "   $(date +%H:%M:%S) $status"
  case "$(echo "$status" | tr '[:upper:]' '[:lower:]')" in
    *complete*) break ;;
    *error* | *cancel* | *fail*) failed=1; break ;;
  esac
  sleep "$poll_sec"
done

mkdir -p "$out_dir"
# the API returns the log as JSON records ({stream_name, time, data}); keep just the text
uv run kaggle kernels logs "$kernel_id" 2>&1 | uv run python -c '
import json, sys
raw = sys.stdin.read()
try:
    print("".join(record.get("data", "") for record in json.loads(raw)), end="")
except ValueError:
    print(raw, end="")
' > "$out_dir/run.log" || true
echo ">> log: $out_dir/run.log"
if [[ -n "${failed:-}" ]]; then
  tail -40 "$out_dir/run.log"
  exit 1
fi

uv run kaggle kernels output "$kernel_id" -p "$out_dir" --file-pattern "$pattern" --force --quiet
rm -f "$out_dir/${kernel_id##*/}.log" # the CLI always adds the raw log; run.log already has it
echo ">> outputs:"
ls -lh "$out_dir"
