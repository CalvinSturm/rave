# RAVE

RAVE is a Rust-native, GPU-resident AI video engine with a bounded decode -> preprocess -> inference -> encode pipeline.

## Workspace Layout

```text
rave/
├── crates/
│   ├── rave-core/
│   ├── rave-cuda/
│   ├── rave-tensorrt/
│   ├── rave-nvcodec/
│   ├── rave-ffmpeg/
│   └── rave-pipeline/
├── rave-cli/
├── examples/
└── Cargo.toml
```

## Build

```bash
cargo build --workspace
```

## Test

```bash
cargo test --workspace
```

## CLI Quickstart

```bash
# Human-readable probe
target/debug/rave probe

# Structured probe JSON
target/debug/rave probe --json

# Probe all devices
target/debug/rave probe --all

# Human-readable benchmark summary
target/debug/rave benchmark --input in.mp4 --model model.onnx --skip-encode

# Structured benchmark JSON on stdout (+ file output)
target/debug/rave benchmark --input in.mp4 --model model.onnx --skip-encode --json --json-out /tmp/bench.json

# Progress (auto prints in TTY; explicit JSONL progress to stderr)
target/debug/rave benchmark --input in.mp4 --model model.onnx --skip-encode --progress jsonl

# Device inventory
target/debug/rave devices --json
```

CLI stdout/stderr + JSON contract (single source of truth):
- `stdout`:
  - default mode (no `--json`): human-readable command summary only
  - `--json` mode: exactly one final JSON object only
- `stderr`:
  - progress stream only when `--progress human|jsonl` or `--jsonl` is set
  - warnings/logging/errors (never mixed into `stdout` JSON mode output)
- Every CLI JSON object includes `"schema_version": 1` in current releases.
- Success payloads include `"ok": true`; failures include `"ok": false` and `"error": "<message>"`.
- Stability promise for parsers:
  - within a fixed `schema_version`, existing field names/types and channel placement (`stdout` final payload, `stderr` progress) remain stable
  - additive fields may be introduced
  - breaking schema or channel-routing changes require incrementing `schema_version`

Progress JSONL contract (`--progress jsonl` or `--jsonl`):
- Stream: `stderr` only (stdout remains human summary or final `--json` payload).
- Cadence: at most 1 line/sec while frame counters change, plus one final line with `final=true`.
- Units: `elapsed_ms` is wall-clock milliseconds; frame counters are cumulative counts.
- Schema:
```json
{"schema_version":1,"type":"progress","command":"benchmark|upscale","elapsed_ms":1234,"frames":{"decoded":120,"inferred":118,"encoded":0},"final":false}
```
- Example lines:
```json
{"schema_version":1,"type":"progress","command":"benchmark","elapsed_ms":1012,"frames":{"decoded":96,"inferred":94,"encoded":0},"final":false}
{"schema_version":1,"type":"progress","command":"benchmark","elapsed_ms":14892,"frames":{"decoded":1421,"inferred":1421,"encoded":0},"final":true}
```

## WSL2 + CUDA + ONNX Runtime TensorRT EP

RAVE can run under WSL2 with NVIDIA GPU acceleration. ONNX Runtime is linked statically in this build, and TensorRT EP provider plugins are loaded dynamically. TensorRT EP requires the provider bridge symbol `Provider_GetHost` from `libonnxruntime_providers_shared.so`.

RAVE now preloads that bridge (`RTLD_GLOBAL`) before TensorRT EP registration, then:
- uses TensorRT EP when registration succeeds
- falls back to CUDA EP with a clear warning when TensorRT EP cannot be loaded

### Required system libraries (WSL)

1. NVIDIA WSL driver libs must be visible (typically `/usr/lib/wsl/lib`).
2. CUDA 12 user-space libs must be resolvable (for example `libcudart.so.12`, `libcublas.so.12`).
3. TensorRT 10 + parser + cuDNN 9 libs must be resolvable (`libnvinfer.so.10`, `libnvonnxparser.so.10`, `libcudnn.so.9`).

Prefer persistent linker config (`ldconfig`) over ad-hoc shell exports.

### Runtime provider control

Use `RAVE_ORT_TENSORRT`:
- `auto` (default): try TensorRT EP, then fallback to CUDA EP on failure
- `on` / `trt-only`: require TensorRT EP (initialization fails if unavailable)
- `off` / `cuda-only`: skip TensorRT EP and use CUDA EP directly

### Verify linkage and startup

```bash
D="$(ls -d ~/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/* | tail -n1)"
ldd -r "$D/libonnxruntime_providers_shared.so"
LD_PRELOAD="$D/libonnxruntime_providers_shared.so" ldd -r "$D/libonnxruntime_providers_tensorrt.so"
nm -D "$D/libonnxruntime_providers_shared.so" | grep Provider_GetHost

cargo run -p rave-cli --bin rave -- \
  upscale \
  --input <input.mp4> \
  --output <output.mp4> \
  --model <model.onnx>
```

Expected startup logs:
- `ORT execution provider selected provider=TensorrtExecutionProvider`, or
- warning about TensorRT EP registration failure followed by `provider=CUDAExecutionProvider`

## Docs

```bash
cargo doc --workspace --no-deps
```

## Examples

```bash
cargo run --example simple_upscale
cargo run --example progress_callback
cargo run --example custom_kernel
cargo run --example batch_directory
cargo run --example benchmark
```

## Prelude

`rave::prelude` re-exports common types and pipeline entry types:

- `GpuTexture`, `FrameEnvelope`, `PixelFormat`
- `PipelineConfig`, `UpscalePipeline`
- `EngineError`, `Result`
