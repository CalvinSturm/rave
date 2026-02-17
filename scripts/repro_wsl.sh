#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/wsl/env.sh"

INPUT="${1:-legacy/engine-v2/test_videos/Input.mp4}"
MODEL="${2:-legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx}"
OUTPUT="${3:-legacy/engine-v2/test_videos/out_repro_wsl.mp4}"

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

  echo "== WSL markers =="
  echo "distro_id=${distro_id}"
  echo "distro_name=${distro_name}"
  if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "/proc/version: microsoft detected"
  else
    echo "/proc/version: microsoft not detected"
  fi
  if [[ -e /dev/dxg ]]; then
    echo "/dev/dxg: present"
  else
    echo "/dev/dxg: missing"
  fi
  echo "note: wsl.exe -l -v is not available inside WSL; run it from Windows PowerShell."
  echo

  if [[ "${distro_id}" == "docker-desktop" || "${distro_name,,}" == *"docker desktop"* ]]; then
    fail "Detected docker-desktop distro. Switch to your Ubuntu WSL distro and rerun."
  fi
  grep -qi microsoft /proc/version 2>/dev/null || fail "This script must run inside WSL2."
  [[ -e /dev/dxg ]] || fail "/dev/dxg is missing; GPU passthrough is unavailable in this distro."
}

export RUST_LOG="${RUST_LOG:-info}"
export RAVE_ORT_TENSORRT="${RAVE_ORT_TENSORRT:-auto}"

resolve_lib() {
  local soname="$1"
  local resolved
  resolved="$(ldconfig -p 2>/dev/null | awk -v n="$soname" '$1 == n { print $NF; exit }')"
  if [[ -n "${resolved}" ]]; then
    echo "${soname} -> ${resolved}"
  else
    echo "${soname} -> not found"
  fi
}

echo "== WSL repro env =="
echo "pwd=$PWD"
echo "input=$INPUT"
echo "model=$MODEL"
echo "output=$OUTPUT"
echo "RAVE_ORT_TENSORRT=$RAVE_ORT_TENSORRT"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo
print_wsl_context
echo "== CUDA/NVENC library resolution =="
resolve_lib "libcuda.so.1"
resolve_lib "libnvcuvid.so.1"
resolve_lib "libnvidia-encode.so.1"
echo
echo "== GPU/driver =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
echo
echo "== ORT provider cache dir =="
find "$HOME/.cache/ort.pyke.io/dfbin" -maxdepth 3 -type d 2>/dev/null | sort | tail -n 1 || true
echo
echo "== Run upscale repro =="
RUN_LOG="$(mktemp /tmp/rave_repro_wsl_XXXX.log)"
set +e
cargo run -p rave-cli --bin rave --locked -- \
  upscale \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model "$MODEL" 2>&1 | tee "$RUN_LOG"
UP_RC=${PIPESTATUS[0]}
set -e

if [[ ${UP_RC} -eq 0 ]]; then
  echo "upscale: PASS"
  exit 0
fi

if grep -Eqi 'nvenc|nvidia-encode|encode session' "$RUN_LOG"; then
  echo
  echo "NVENC failure detected. Falling back to benchmark mode with encode skipped."
  if [[ "$OUTPUT" == *.* ]]; then
    BENCH_JSON="${OUTPUT%.*}.benchmark.json"
  else
    BENCH_JSON="${OUTPUT}.benchmark.json"
  fi

  cargo run -p rave-cli --bin rave --locked -- \
    benchmark \
    --input "$INPUT" \
    --model "$MODEL" \
    --skip-encode \
    --json-out "$BENCH_JSON"

  echo "fallback: PASS (benchmark JSON at $BENCH_JSON)"
  exit 0
fi

echo "upscale: FAIL (exit ${UP_RC}); log=${RUN_LOG}"
exit "${UP_RC}"
