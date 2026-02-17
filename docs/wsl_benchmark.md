# WSL Manual Benchmark (No Network Calls by Agent)

Run these commands directly in your own WSL terminal from repo root:

```bash
cd "/mnt/c/Users/Calvin/Software Projects/rave"
mkdir -p reports
LOG="reports/wsl_bench_$(date +%F_%H%M%S).log"

echo "Phase10 WSL bench started: $(date --iso-8601=seconds)" | tee "$LOG"
# shellcheck disable=SC1091
source scripts/wsl/env.sh

RUST_LOG=info cargo run -p rave-cli --bin rave --locked -- \
  benchmark \
  --input legacy/engine-v2/test_videos/Input.mp4 \
  --model legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx \
  --skip-encode \
  --json-out /tmp/bench.json 2>&1 | tee -a "$LOG"

echo "Phase10 WSL bench finished: $(date --iso-8601=seconds)" | tee -a "$LOG"
echo "Benchmark JSON: /tmp/bench.json" | tee -a "$LOG"
echo "Log written to: $LOG"
```

Equivalent helper script (same behavior):

```bash
bash scripts/wsl/run_bench.sh
```
