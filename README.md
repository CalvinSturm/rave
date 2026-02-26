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
│   ├── rave-pipeline/
│   └── rave-runtime-nvidia/
├── rave-cli/
└── Cargo.toml
```

## Crate Docs

- `crates/rave-core/README.md`
- `crates/rave-cuda/README.md`
- `crates/rave-ffmpeg/README.md`
- `crates/rave-nvcodec/README.md`
- `crates/rave-pipeline/README.md`
- `crates/rave-runtime-nvidia/README.md`
- `crates/rave-tensorrt/README.md`
- `rave-cli/README.md`

## Architecture Boundaries

Allowed internal dependency graph (mechanically checked by `./scripts/check_deps.sh`):

```text
rave-core      -> (no internal deps)
rave-cuda      -> rave-core
rave-tensorrt  -> rave-core (optionally rave-cuda utilities)
rave-nvcodec   -> rave-core (optionally rave-cuda utilities)
rave-ffmpeg    -> rave-core
rave-pipeline       -> rave-core, rave-cuda, rave-tensorrt (+ optional graph runtime features)
rave-runtime-nvidia -> rave-core, rave-cuda, rave-tensorrt, rave-nvcodec, rave-ffmpeg, rave-pipeline
rave-cli            -> rave-core, rave-pipeline, rave-runtime-nvidia
```

Why these boundaries exist:
- Keep `rave-core` portable and free of engine wiring concerns.
- Keep leaf crates focused on one domain (CUDA, TensorRT, codec, container I/O).
- Keep generic orchestration in `rave-pipeline`.
- Keep concrete backend composition in `rave-runtime-nvidia`.

Feature placement decision table:
- Shared types/traits/errors: `rave-core`
- CUDA kernels/utilities: `rave-cuda`
- Inference runtime behavior: `rave-tensorrt`
- Decode/encode hardware behavior: `rave-nvcodec`
- Container demux/mux/probe behavior: `rave-ffmpeg`
- Multi-stage orchestration and graph/runtime contracts: `rave-pipeline`
- Concrete NVIDIA runtime composition (CUDA + TensorRT + FFmpeg + NVDEC/NVENC): `rave-runtime-nvidia`
- CLI argument parsing/output formatting: `rave-cli`

No-host-copies checklist:
- `docs/no_host_copies.md`

Unsafe-boundary audit checklist:
- `docs/unsafe_audit.md`

Consumer integration contract:
- `docs/integration_contract.md`
- `docs/windows_integration_checklist.md` (Windows build/test/runtime)

Stage graph integration API:
- `rave-pipeline` exports `StageGraph`, `StageConfig`, `ProfilePreset`,
  `RunContract`, and `UpscalePipeline::run_graph(...)`.
- `rave-runtime-nvidia` exports concrete runtime setup helpers (`prepare_runtime`,
  `resolve_input`, `create_decoder`, `create_nvenc_encoder`).
- `ProfilePreset::ProductionStrict` enables strict no-host-copies policy and
  deterministic contract checks (via optional checkpoint hooks).
- Graph specs are schema-versioned and must include
  `"graph_schema_version": 1` at the top level.

## Build

```bash
cargo build --workspace
```

Windows (PowerShell):

```powershell
.\scripts\build.ps1 --workspace
.\scripts\build.ps1 -- -p rave-cli --bin rave --release
```

## Test

```bash
cargo test --workspace
```

Windows (PowerShell):

```powershell
.\scripts\test.ps1
```

Runtime smoke (Windows PowerShell):

```powershell
.\scripts\smoke_upscale.ps1
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

# Best-effort runtime validation harness
target/debug/rave validate --json --best-effort

# Fixture-driven validation scenario (self-contained default model)
target/debug/rave validate --json --best-effort --profile production_strict \
  --fixture tests/fixtures/validate_production_strict.json

# Optional model override
RAVE_VALIDATE_MODEL=/abs/path/custom_model.onnx \
  target/debug/rave validate --json --best-effort --profile production_strict \
  --fixture tests/fixtures/validate_production_strict.json
```

`validate --fixture tests/fixtures/validate_production_strict.json` resolves the
model in this order:
1. `RAVE_VALIDATE_MODEL` (if set)
2. committed fallback `tests/assets/models/resize2x_rgb.onnx`

Validate JSON output now includes `"model_path"` so runs are transparent even
when best-effort mode skips due missing runtime dependencies.

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
{"schema_version":1,"type":"progress","command":"benchmark|upscale|validate","elapsed_ms":1234,"frames":{"decoded":120,"inferred":118,"encoded":0},"final":false}
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

## Integration Entry Points

Use the real entry points that exist today:

- CLI workflows: `rave-cli` (`rave probe`, `rave devices`, `rave benchmark`, `rave upscale`, `rave validate`)
- Library orchestration: `rave-pipeline` exports `UpscalePipeline`, `PipelineConfig`, `StageGraph`, `ProfilePreset`, `RunContract`
- Concrete runtime composition: `rave-runtime-nvidia` exports runtime setup and codec/container wiring helpers
- Shared contracts: `rave-core` exports `GpuTexture`, `FrameEnvelope`, `PixelFormat`, `EngineError`, `Result`
