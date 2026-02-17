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

echo "== WSL detection =="
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  echo "distro_id=${ID:-unknown}"
  echo "distro_name=${PRETTY_NAME:-${NAME:-unknown}}"
fi
if [[ -e /dev/dxg ]]; then
  echo "/dev/dxg: present"
else
  echo "/dev/dxg: missing"
fi
if grep -qi microsoft /proc/version; then
  echo "/proc/version: microsoft detected"
else
  echo "/proc/version: microsoft not detected"
fi
echo "note: wsl.exe -l -v is not available inside WSL; run it from Windows PowerShell."
if [[ "${ID:-}" == "docker-desktop" || "${PRETTY_NAME:-}" =~ [Dd]ocker[[:space:]]Desktop ]]; then
  fail "Detected docker-desktop distro. Switch to your Ubuntu WSL distro and rerun."
fi
grep -qi microsoft /proc/version 2>/dev/null || fail "This script must run inside WSL2."
[[ -e /dev/dxg ]] || fail "/dev/dxg is missing; GPU passthrough is unavailable in this distro."
echo

echo "== kernel =="
cat /proc/version
echo

echo "== nvidia-smi (first 30 lines) =="
nvidia-smi 2>&1 | sed -n '1,30p' || true
echo

echo "== ldconfig sanity =="
ldconfig -p | grep -E 'libcuda|libcudart|libcublas' || true
echo

[[ -f /usr/lib/wsl/lib/libcuda.so.1 ]] || fail "/usr/lib/wsl/lib/libcuda.so.1 is missing"

echo "== ldd /usr/lib/wsl/lib/libcuda.so.1 =="
LDD_OUT="$(ldd /usr/lib/wsl/lib/libcuda.so.1 2>&1 || true)"
echo "${LDD_OUT}"
if echo "${LDD_OUT}" | grep -q "not found"; then
  fail "ldd reported missing dependencies for /usr/lib/wsl/lib/libcuda.so.1"
fi
echo

echo "== forbidden package scan =="
FORBIDDEN="$(dpkg -l 2>/dev/null | awk '/^ii/ {print $2}' | grep -E '^(nvidia-driver-.*|cuda-drivers|cuda|cuda-12-.*)$' || true)"
if [[ -n "${FORBIDDEN}" ]]; then
  echo "WARNING: Detected Linux GPU driver/toolkit meta-packages inside WSL:"
  echo "${FORBIDDEN}"
  echo "Suggested purge:"
  echo "  sudo apt-get purge -y 'nvidia-driver-*' 'cuda-drivers' 'cuda' 'cuda-12-*'"
  echo "  sudo apt-get autoremove -y"
else
  echo "No forbidden packages detected."
fi
echo

echo "== cuda probe =="
PROBE_LOG="$(mktemp /tmp/rave_cuda_probe_XXXX.log)"
set +e
scripts/run_cuda_probe.sh >"${PROBE_LOG}" 2>&1
PROBE_RC=$?
set -e
cat "${PROBE_LOG}"

if ! grep -q "cuInit rc=0" "${PROBE_LOG}"; then
  fail "cuda probe shows cuInit != 0. See ${PROBE_LOG}"
fi
if [[ ${PROBE_RC} -ne 0 ]]; then
  fail "scripts/run_cuda_probe.sh exited ${PROBE_RC}. See ${PROBE_LOG}"
fi

echo
echo "WSL GPU healthcheck: PASS"
