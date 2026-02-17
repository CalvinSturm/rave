#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/wsl/env.sh"

INPUT="${1:-legacy/engine-v2/test_videos/Input.mp4}"
MODEL="${2:-legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx}"
OUTPUT="${3:-legacy/engine-v2/test_videos/out_diag_wsl_cuda.mp4}"

echo "== env =="
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "LIBRARY_PATH=${LIBRARY_PATH:-}"
echo

if [[ ! -x target/debug/rave ]]; then
  echo "building target/debug/rave ..."
  cargo build -p rave-cli --locked >/dev/null
fi

LOG="$(mktemp /tmp/rave_ld_debug_libs_XXXX.log)"
echo "== LD_DEBUG=libs probe (capturing to ${LOG}) =="
set +e
LD_DEBUG=libs target/debug/rave \
  upscale \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" >"$LOG" 2>&1
RC=$?
set -e

grep -i "libcuda" "$LOG" || true
echo
if grep -q "/usr/lib/wsl/lib/libcuda.so.1" "$LOG"; then
  echo "PASS: libcuda resolved from /usr/lib/wsl/lib/libcuda.so.1"
else
  echo "FAIL: libcuda did not resolve from /usr/lib/wsl/lib/libcuda.so.1"
fi

echo "rave_exit_code=${RC}"
echo "full_log=${LOG}"
