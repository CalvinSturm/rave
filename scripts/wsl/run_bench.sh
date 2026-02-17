#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/wsl/env.sh"

INPUT="${1:-legacy/engine-v2/test_videos/Input.mp4}"
MODEL="${2:-legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx}"
JSON_OUT="${3:-/tmp/bench.json}"
LOG="reports/wsl_bench_$(date +%F_%H%M%S).log"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

print_wsl_context() {
  local distro_id="unknown"
  local distro_name="unknown"
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    distro_id="${ID:-unknown}"
    distro_name="${PRETTY_NAME:-${NAME:-unknown}}"
  fi

  echo "distro_id=${distro_id}" | tee -a "$LOG"
  echo "distro_name=${distro_name}" | tee -a "$LOG"
  if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "/proc/version: microsoft detected" | tee -a "$LOG"
  else
    echo "/proc/version: microsoft not detected" | tee -a "$LOG"
  fi
  if [[ -e /dev/dxg ]]; then
    echo "/dev/dxg: present" | tee -a "$LOG"
  else
    echo "/dev/dxg: missing" | tee -a "$LOG"
  fi
  echo "note: wsl.exe -l -v is not available inside WSL; run it from Windows PowerShell." | tee -a "$LOG"

  if [[ "${distro_id}" == "docker-desktop" || "${distro_name,,}" == *"docker desktop"* ]]; then
    fail "Detected docker-desktop distro. Switch to your Ubuntu WSL distro and rerun."
  fi
  grep -qi microsoft /proc/version 2>/dev/null || fail "This script must run inside WSL2."
  [[ -e /dev/dxg ]] || fail "/dev/dxg is missing; GPU passthrough is unavailable in this distro."
}

mkdir -p reports
export RUST_LOG="${RUST_LOG:-info}"

echo "Phase10 WSL benchmark started: $(date --iso-8601=seconds)" | tee "$LOG"
echo "Repo: $ROOT_DIR" | tee -a "$LOG"
echo "Input: $INPUT" | tee -a "$LOG"
echo "Model: $MODEL" | tee -a "$LOG"
echo "JSON_OUT: $JSON_OUT" | tee -a "$LOG"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | tee -a "$LOG"
echo | tee -a "$LOG"

echo "== WSL detection ==" | tee -a "$LOG"
print_wsl_context
echo | tee -a "$LOG"

echo "== Benchmark run ==" | tee -a "$LOG"
cargo run -p rave-cli --bin rave --locked -- \
  benchmark \
  --input "$INPUT" \
  --model "$MODEL" \
  --skip-encode \
  --json-out "$JSON_OUT" 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "Phase10 WSL benchmark finished: $(date --iso-8601=seconds)" | tee -a "$LOG"
echo "JSON written to: $JSON_OUT" | tee -a "$LOG"
echo "Log written to: $LOG" | tee -a "$LOG"
