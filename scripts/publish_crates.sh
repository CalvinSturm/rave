#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" || ( "${MODE}" != "--dry-run" && "${MODE}" != "--publish" ) ]]; then
  echo "usage: $0 --dry-run|--publish" >&2
  exit 2
fi

if [[ "${MODE}" == "--publish" && -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  echo "CARGO_REGISTRY_TOKEN must be set for --publish" >&2
  exit 2
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required" >&2
  exit 2
fi

CRATES=(
  "rave-core"
  "rave-cuda"
  "rave-ffmpeg"
  "rave-nvcodec"
  "rave-tensorrt"
  "rave-pipeline"
  "rave-cli"
)

PUBLISH_ARGS=("--locked")
if [[ "${MODE}" == "--dry-run" ]]; then
  PUBLISH_ARGS+=("--dry-run" "--allow-dirty")
fi

echo "Publishing mode: ${MODE}"
echo "Crate order: ${CRATES[*]}"

for crate in "${CRATES[@]}"; do
  echo
  echo "==> cargo publish -p ${crate} ${PUBLISH_ARGS[*]}"
  cargo publish -p "${crate}" "${PUBLISH_ARGS[@]}"

  # crates.io index propagation can lag for dependents; be conservative.
  if [[ "${MODE}" == "--publish" ]]; then
    sleep 20
  fi
done

echo
echo "Done."
