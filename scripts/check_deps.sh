#!/usr/bin/env bash
set -euo pipefail

python3 scripts/check_deps.py --self-test
cargo metadata --format-version 1 --no-deps | python3 scripts/check_deps.py
