#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/wsl/env.sh"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  echo "distro_id=${ID:-unknown}"
  echo "distro_name=${PRETTY_NAME:-${NAME:-unknown}}"
fi
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

if [[ "${ID:-}" == "docker-desktop" || "${PRETTY_NAME:-}" =~ [Dd]ocker[[:space:]]Desktop ]]; then
  fail "Detected docker-desktop distro. Switch to your Ubuntu WSL distro and rerun."
fi
grep -qi microsoft /proc/version 2>/dev/null || fail "This script must run inside WSL2."
[[ -e /dev/dxg ]] || fail "/dev/dxg is missing; GPU passthrough is unavailable in this distro."

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

cargo run -p rave-cli --bin cuda_probe --locked
