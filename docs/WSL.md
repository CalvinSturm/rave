# WSL2 Setup (CUDA + ORT TensorRT EP + NVENC)

This project is tested on WSL2 Ubuntu x86_64 with an NVIDIA GPU.

## Prerequisites

1. WSL2 NVIDIA driver support enabled:
   - `/usr/lib/wsl/lib/libcuda.so.1` exists
   - `nvidia-smi` works in WSL
2. CUDA toolkit runtime libraries installed (preferred path):
   - `/usr/local/cuda-12/targets/x86_64-linux/lib`
3. TensorRT + parser + cuDNN available via dynamic loader:
   - `libnvinfer.so.10`
   - `libnvonnxparser.so.10`
   - `libcudnn.so.9`

Do not rely on unrelated software installs (for example Ollama) to provide `libcudart.so.12`.

## Recommended runtime env

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
```

`rave-tensorrt` also preloads CUDA runtime libs from `/usr/local/cuda-12/targets/x86_64-linux/lib` (and `/usr/local/cuda/lib64`) before ORT TensorRT EP registration.

## ORT TensorRT EP loading model

RAVE links ONNX Runtime core statically and uses dynamic EP provider `.so` files.
For TensorRT EP, `libonnxruntime_providers_shared.so` must be globally loaded before `libonnxruntime_providers_tensorrt.so` so `Provider_GetHost` resolves.

RAVE now does this explicitly on Linux/WSL:

1. `dlopen(libonnxruntime_providers_shared.so, RTLD_NOW | RTLD_GLOBAL)`
2. Resolve provider directory/path deterministically and configure ORT provider registration
3. Register TensorRT EP in ORT (without explicit provider constructor preload)

If TensorRT EP fails in `RAVE_ORT_TENSORRT=auto`, RAVE logs the reason and falls back to CUDA EP.

At startup RAVE now resolves and logs:

- `libonnxruntime_providers_shared.so`
- `libonnxruntime_providers_tensorrt.so` or `libonnxruntime_providers_cuda.so`

Search order is deterministic:

1. `ORT_DYLIB_PATH`
2. `ORT_LIB_LOCATION`
3. next to the executable (`target/debug`, `deps`)
4. newest `~/.cache/ort.pyke.io/dfbin/**`

On WSL, after env vars, RAVE prefers the newest ORT cache directory before executable-adjacent paths.

## Repro command

```bash
scripts/repro_wsl.sh \
  legacy/engine-v2/test_videos/Input.mp4 \
  legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx \
  legacy/engine-v2/test_videos/out_repro_wsl.mp4
```

## Direct CLI command (new syntax)

```bash
cargo run -p rave-cli --bin rave --locked -- \
  upscale \
  --input legacy/engine-v2/test_videos/Input.mp4 \
  --output legacy/engine-v2/test_videos/out_repro_wsl.mp4 \
  --model legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx
```

## Benchmark command (skip encode, JSON output)

```bash
target/debug/rave benchmark \
  --input legacy/engine-v2/test_videos/Input.mp4 \
  --model legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx \
  --skip-encode \
  --json-out /tmp/bench.json
```

## Example: 10-line GPU video upscale

```bash
#!/usr/bin/env bash
set -euo pipefail
IN=legacy/engine-v2/test_videos/Input.mp4
OUT=legacy/engine-v2/test_videos/out_10line.mp4
MODEL=legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
cargo run -p rave-cli --bin rave --locked -- upscale \
  --input "$IN" --output "$OUT" --model "$MODEL" \
  --precision fp32 --device 0
```

## Troubleshooting

1. Verify ORT provider deps:
```bash
scripts/test_ort_provider_load.sh
```

2. Force CUDA EP only (skip TensorRT EP):
```bash
RAVE_ORT_TENSORRT=off scripts/repro_wsl.sh
```

3. Force TensorRT EP only:
```bash
RAVE_ORT_TENSORRT=on scripts/repro_wsl.sh
```

4. Capture benchmark crash backtrace under gdb:
```bash
scripts/gdb_bench.sh
cat /tmp/rave_gdb_bt.txt
```

5. Capture latest systemd-coredump stack trace:
```bash
scripts/coredump_last.sh
```

## CUDA_ERROR_OPERATING_SYSTEM (304) on cuInit

Runbook (command-first):

```bash
wsl --shutdown
wsl --update
# reboot Windows
```

Then update the Windows NVIDIA driver to the latest WSL-supported release.

Do not install Linux NVIDIA driver stacks inside WSL (`nvidia-driver-*`, `cuda-drivers`).

Run:

```bash
scripts/wsl_gpu_healthcheck.sh
scripts/run_cuda_probe.sh
```

If logs show `/usr/lib/wsl/drivers/.../libcuda.so.1.1` during failure, prepend:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
```
