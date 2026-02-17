#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/wsl/env.sh"

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <input_video> <output_video> <model.onnx> [trt_mode:auto|on|off]"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"
MODEL="$3"
TRT_MODE="${4:-auto}"

D="$(find "$HOME/.cache/ort.pyke.io/dfbin" -maxdepth 3 -type d | sort | tail -n1)"
if [[ -z "${D}" || ! -d "${D}" ]]; then
  echo "failed to find ORT cache directory under ~/.cache/ort.pyke.io/dfbin"
  exit 1
fi

echo "ORT cache dir: ${D}"
echo
echo "[1/4] ldd -r libonnxruntime_providers_shared.so"
ldd -r "${D}/libonnxruntime_providers_shared.so"
echo
echo "[2/4] ldd -r libonnxruntime_providers_tensorrt.so"
LD_PRELOAD="${D}/libonnxruntime_providers_shared.so" ldd -r "${D}/libonnxruntime_providers_tensorrt.so"
echo
echo "[3/4] Provider_GetHost export check"
nm -D "${D}/libonnxruntime_providers_shared.so" | grep Provider_GetHost
echo
echo "[4/4] RAVE smoke run (RAVE_ORT_TENSORRT=${TRT_MODE})"
RUST_LOG=info RAVE_ORT_TENSORRT="${TRT_MODE}" cargo run -p rave-cli --bin rave -- \
  upscale \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --model "${MODEL}"
