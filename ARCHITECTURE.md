# RAVE — Architecture

## Workspace crate boundaries

RAVE enforces internal crate dependency boundaries via `cargo metadata` checks
(`./scripts/check_deps.sh`, wired into CI).

Allowed edges:

```text
rave-core      -> (no internal deps)
rave-cuda      -> rave-core
rave-tensorrt  -> rave-core (optional rave-cuda utilities)
rave-nvcodec   -> rave-core (optional rave-cuda utilities)
rave-ffmpeg    -> rave-core
rave-pipeline       -> rave-core, cudarc (+ optional rave-cuda/rave-tensorrt/rave-nvcodec/rave-ffmpeg)
rave-runtime-nvidia -> rave-core, rave-cuda, rave-tensorrt, rave-nvcodec, rave-ffmpeg
rave-cli            -> rave-core, rave-pipeline, rave-runtime-nvidia
```

Boundary rationale:
- `rave-core` is the neutral type/trait/error layer.
- Domain crates (`rave-cuda`, `rave-tensorrt`, `rave-nvcodec`, `rave-ffmpeg`) stay focused.
- `rave-pipeline` owns generic orchestration and graph contracts.
- `rave-runtime-nvidia` owns concrete NVIDIA backend composition.

Decision table for new features:
- Shared contracts and reusable primitives -> `rave-core`
- Device kernels and stream helpers -> `rave-cuda`
- ORT/TensorRT execution behavior -> `rave-tensorrt`
- NVDEC/NVENC codec behavior -> `rave-nvcodec`
- Container I/O and packet boundaries -> `rave-ffmpeg`
- Generic stage orchestration and strict runtime contracts -> `rave-pipeline`
- Concrete NVIDIA stack wiring (CUDA + TensorRT + FFmpeg + NVDEC/NVENC) -> `rave-runtime-nvidia`
- CLI UX/contracts -> `rave-cli`

No-host-copies checklist:
- `docs/no_host_copies.md`

## Stage graph API

`rave-pipeline` provides a stable orchestration surface for multi-app reuse:

- `StageGraph` (v1 linear chain)
- `StageConfig` / `StageKind` (`Enhance`)
- `ProfilePreset` (`Dev`, `ProductionStrict`, `Benchmark`)
- `RunContract` (determinism + audit policy)
- `UpscalePipeline::run_graph(input, output, graph, profile, contract)`
- Graph schema contract: top-level `graph_schema_version` is required and must
  equal `1` for current builds.

v1 constraints:
- at least one stage
- exactly one `Enhance` stage

This keeps behavior deterministic and mechanically auditable while leaving room
for a future DAG scheduler without breaking the public contract.

## Production strict profile

`ProfilePreset::ProductionStrict` is the first production profile and enforces:
- strict no-host-copies mode (`PipelineConfig.strict_no_host_copies=true`)
- hard-fail on audit/invariant warnings
- deterministic output contract checks at canonical stage boundaries

Container bytes are not the determinism boundary; canonical stage checkpoint
hashes are used for repeat-run comparison when enabled.

## Validate fixture runtime

`rave validate --fixture tests/fixtures/validate_production_strict.json` is
designed to run end-to-end on GPU runners without manual model inputs.

Model resolution order for this fixture:
1. `RAVE_VALIDATE_MODEL` (explicit override)
2. committed fallback `tests/assets/models/resize2x_rgb.onnx`

This keeps CI deterministic while preserving local override flexibility.

## Pipeline overview

RAVE runs four concurrent tasks connected by bounded async channels. Frame data stays in GPU device memory from decode to encode — no `cudaMemcpy` in steady-state operation.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              RAVE Pipeline                                  │
│                                                                              │
│  ┌──────────┐  ch(4)  ┌────────────┐  ch(2)  ┌───────────┐  ch(4)  ┌──────────┐
│  │ Decoder  │────────►│ Preprocess │────────►│ Inference │────────►│ Encoder  │
│  │ (NVDEC)  │         │(CUDA kern) │         │(TensorRT) │         │ (NVENC)  │
│  │ blocking │         │   async    │         │   async   │         │ blocking │
│  └──────────┘         └────────────┘         └───────────┘         └──────────┘
│       │                     │                      │                     │
│  NV12 (GPU)           RGB F32/F16            RGB F32/F16           NV12 (GPU)
│                        NCHW planar           NCHW planar
│                                                                              │
│  ═══════════ GPU-resident frame transport · no CPU frame copies ════════════│
└──────────────────────────────────────────────────────────────────────────────┘
```

**Stage execution model:**

- **Decode** and **Encode** run as `tokio::task::spawn_blocking` — NVDEC/NVENC issue synchronous CUDA driver calls that would block the async executor.
- **Preprocess** and **Inference** run as regular `tokio::spawn` tasks — they issue async CUDA kernel launches and await results via CUDA stream synchronization.
- End-of-stream is signaled by channel closure (sender drop), not sentinel values.

## Backpressure

Channel capacities (4, 2, 4) enforce bounded VRAM usage. Backpressure flows upstream via `.send().await` suspension:

```
Encoder stalls
  → inference→encode channel fills (cap 4)
    → inference stage's send().await blocks
      → preprocess→inference channel fills (cap 2)
        → preprocess stage's send().await blocks
          → decode→preprocess channel fills (cap 4)
            → decoder stalls
```

**Bound:** Total in-flight frames ≤ `sum(channel_capacities)` = 10 frames. At 4K RGB F32, that's ~10 × 95 MB ≈ 950 MB of VRAM committed to in-flight data, independent of video length.

Channel capacities are configurable via `PipelineConfig`:

```rust
pub struct PipelineConfig {
    pub decoded_capacity: usize,      // default: 4
    pub preprocessed_capacity: usize,  // default: 2
    pub upscaled_capacity: usize,      // default: 4
    pub encoder_nv12_pitch: usize,
    pub model_precision: ModelPrecision,
    pub enable_profiler: bool,
    pub strict_no_host_copies: bool,   // default: false
}
```

Queue depths are tracked at runtime via `QueueDepthTracker` (lock-free `AtomicUsize` per stage) and exposed in `HealthSnapshot` for telemetry.

## Micro-batching

The TensorRT backend defines `BatchConfig` for planned micro-batching
(not yet implemented — current inference is always single-frame):

```rust
pub struct BatchConfig {
    pub max_batch: usize,         // default: 1 (single-frame)
    pub latency_deadline_us: u64, // default: 8000 (8 ms — half a 60fps frame)
}
```

Current status: `BatchConfig` is API-plumbed, but runtime execution scheduling is
single-frame today (`N=1` dispatch in the active `process()` path). There is no
batch queue or latency-deadline flush loop implemented yet.

To implement real micro-batching, the inference stage would need:
- a bounded batch queue/collector in front of backend dispatch
- a flush policy driven by `latency_deadline_us`
- batch-aware output ordering and error handling

## Crate dependency graph (workspace reality)

```
rave-cli
  ├── rave-core        (shared errors/types used by CLI contracts)
  ├── rave-pipeline    (generic orchestration + graph schema)
  │     ├── rave-core
  │     ├── rave-cuda
  │     └── rave-tensorrt
  └── rave-runtime-nvidia (concrete NVIDIA runtime composition)
        ├── rave-core
        ├── rave-cuda
        ├── rave-tensorrt
        ├── rave-nvcodec
        ├── rave-ffmpeg
        └── rave-pipeline

Leaf domain crates:
- rave-cuda      -> rave-core
- rave-tensorrt  -> rave-core (rave-cuda allowed by policy, not required today)
- rave-nvcodec   -> rave-core (rave-cuda allowed by policy, not required today)
- rave-ffmpeg    -> rave-core
- rave-core      -> (no internal rave-* deps)
```

## Data flow

Each frame passes through four pixel format stages:

| Stage | Format | Layout | Details |
|-------|--------|--------|---------|
| NVDEC output | NV12 | Y plane + interleaved UV plane | Semi-planar 4:2:0, 8-bit. Pitch may exceed width for alignment. |
| After preprocess | RGB F32 planar | 3 × H × W × 4 bytes | NCHW layout, values normalized to [0, 1]. FP16 variant uses `half::f16`. |
| After inference | RGB F32 planar | 3 × (H×scale) × (W×scale) × 4 bytes | Upscaled by the model's scale factor (typically 2× or 4×). |
| After postprocess | NV12 | Y plane + interleaved UV plane | Converted back from RGB for NVENC. Upscaled resolution. |

The preprocess and postprocess conversions are performed by CUDA kernels compiled at startup via NVRTC (cudarc). The kernels handle NV12↔RGB conversion, normalization, pitch alignment, and chroma subsampling. Kernel execution time is tracked per-launch via `StageMetrics`.

## Memory model

### Bucketed buffer pool

`GpuContext` owns a `BucketedPool` that recycles device memory allocations:

- **Buckets** are keyed by allocation size. When a stage needs a buffer, it requests from the pool. If a matching-size buffer exists, it's reused; otherwise a new `CudaSlice<u8>` is allocated from the CUDA driver.
- **Recycling**: When a `GpuTexture`'s `Arc` refcount reaches zero, its backing `CudaSlice` is returned to the pool rather than freed.
- **Warm-up phase**: The first few frames cause fresh allocations. After warm-up, the pool satisfies all requests from recycled buffers — zero CUDA driver allocations in steady state.
- **Pool statistics**: `PoolStats` uses atomic counters to track hits, misses, recycles, and overflows. Reported at pipeline completion and exposed in `HealthSnapshot`.

### VRAM accounting

`GpuContext` maintains an atomic byte counter tracking total device memory in
use. An optional `--vram-limit` flag sets a ceiling for observability, but the
allocator currently warns when usage exceeds the configured limit; it does not
return a dedicated VRAM-limit error variant at allocation time.

### Ownership model

- `GpuTexture.data` is `Arc<CudaSlice<u8>>` — reference-counted, `Send + Sync`, RAII. A compile-time `assert_send_sync::<GpuTexture>()` enforces the bound.
- `FrameEnvelope` wraps a `GpuTexture` with metadata (frame index, PTS, keyframe flag). This is the unit that flows through channels.
- `Arc` allows cheap cloning (refcount increment, no device memory copy) when a frame is referenced by multiple stages simultaneously.
- The `OutputRing` in the TensorRT backend pre-allocates a fixed ring of output buffers. `RingMetrics` tracks slot reuse vs. contention (strong_count > 1 at acquire time).

### Health snapshot

`HealthSnapshot` provides an immutable point-in-time view of engine state:

```rust
pub struct HealthSnapshot {
    pub vram_current_bytes: usize,
    pub vram_peak_bytes: usize,
    pub vram_limit_bytes: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub pool_hit_rate: f64,
    pub pool_overflows: usize,
    pub steady_state: bool,         // true when pool_hit_rate > threshold
    pub decode_queue_depth: usize,
    pub preprocess_queue_depth: usize,
    pub inference_queue_depth: usize,
}
```

## Concurrency model

The pipeline uses `tokio` for async orchestration:

```
                    tokio runtime (multi-threaded)
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  spawn_blocking(decode_loop)                            │
    │       │                                                 │
    │       ▼ mpsc(4)                                         │
    │  spawn(preprocess_loop)                                 │
    │       │                                                 │
    │       ▼ mpsc(2)                                         │
    │  spawn(inference_loop)                                  │
    │       │                                                 │
    │       ▼ mpsc(4)                                         │
    │  spawn_blocking(encode_loop)                            │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

### CUDA streams

`GpuContext` manages three dedicated CUDA streams:

| Stream | Purpose | Hardware unit |
|--------|---------|---------------|
| `decode_stream` | NVDEC decode operations | Video decoder ASIC |
| `preprocess_stream` | NV12↔RGB kernel launches | Compute SMs |
| `inference_stream` | TensorRT model execution | Compute SMs (+ Tensor Cores) |

NVENC uses its own internal stream managed by the encoder API.

### Cross-stream synchronization

CUDA events synchronize handoffs between stages operating on different streams. Before preprocessing begins on a decoded frame, a CUDA event recorded on `decode_stream` is waited on `preprocess_stream`, ensuring the decode is complete without blocking the CPU.

### Graceful shutdown

A `tokio_util::CancellationToken` propagates Ctrl+C (via `tokio::signal`) to all stages. Each stage checks the token between frames and exits cleanly. Channel closure cascades downstream to signal EOS.

## Telemetry

Telemetry is collected via lock-free atomic counters with no mutex contention on the hot path:

| Metric type | Struct | Counters |
|-------------|--------|----------|
| Pipeline throughput | `PipelineMetrics` | frames decoded/preprocessed/inferred/encoded, per-stage latency accumulators (μs) |
| Inference performance | `InferenceMetrics` | total frames, cumulative inference time, peak single-frame time |
| Kernel timing | `StageMetrics` | total kernel execution time (ms), launch count |
| Output ring | `RingMetrics` | slot reuse count, contention events, first-use count |
| System health | `HealthSnapshot` | VRAM current/peak/limit, pool hit rate, queue depths, steady-state flag |

All metrics are reported via `tracing` at pipeline completion. `PipelineMetrics::validate()` asserts the monotonic invariant `decoded ≥ preprocessed ≥ inferred ≥ encoded`.
