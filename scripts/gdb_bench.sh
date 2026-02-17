#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/wsl/env.sh"

INPUT="${1:-legacy/engine-v2/test_videos/Input.mp4}"
MODEL="${2:-legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx}"
JSON_OUT="${3:-/tmp/bench.json}"
BT_OUT="${4:-/tmp/rave_gdb_bt.txt}"

if [[ ! -x target/debug/rave ]]; then
  cargo build -p rave-cli --bin rave --locked
fi

echo "Running benchmark under gdb; backtrace output -> ${BT_OUT}"
gdb -q --batch \
  -ex "set pagination off" \
  -ex "run" \
  -ex "bt" \
  -ex "thread apply all bt" \
  --args target/debug/rave benchmark \
    --input "$INPUT" \
    --model "$MODEL" \
    --skip-encode \
    --json-out "$JSON_OUT" >"$BT_OUT" 2>&1 || true

echo "Saved gdb output to ${BT_OUT}"
