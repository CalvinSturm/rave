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

wait_for_crates_io_version() {
  local crate="$1"
  local version="$2"
  local tries=30
  local sleep_s=5

  for ((i = 1; i <= tries; i++)); do
    local line
    line="$(cargo search "${crate}" --limit 1 2>/dev/null | head -n1 || true)"
    if [[ "${line}" == "${crate} = \"${version}\""* ]]; then
      return 0
    fi
    sleep "${sleep_s}"
  done

  return 1
}

echo "Publishing mode: ${MODE}"
echo "Crate order: ${CRATES[*]}"

for crate in "${CRATES[@]}"; do
  echo
  if [[ "${MODE}" == "--dry-run" ]]; then
    # NOTE: publish --dry-run fails for interdependent bumped versions because
    # crates.io does not yet have the new dependency versions. package gives us
    # local tarball + metadata + verify checks without requiring staged uploads.
    echo "==> cargo package -p ${crate} --locked --allow-dirty"
    cargo package -p "${crate}" --locked --allow-dirty
  else
    local_version="$(cargo pkgid -p "${crate}" | sed -E 's/.*@([^ ]+)$/\1/')"
    echo "==> cargo publish -p ${crate} --locked"
    cargo publish -p "${crate}" --locked
    echo "Waiting for crates.io index to show ${crate} ${local_version}..."
    if ! wait_for_crates_io_version "${crate}" "${local_version}"; then
      echo "Timed out waiting for crates.io index to publish ${crate} ${local_version}" >&2
      exit 1
    fi
  fi
done

echo
echo "Done."
