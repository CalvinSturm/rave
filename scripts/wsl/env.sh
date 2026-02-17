#!/usr/bin/env bash
set -euo pipefail

if grep -qi microsoft /proc/version 2>/dev/null; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi
