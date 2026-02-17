# RAVE v0.1 — Comprehensive Technical Audit

**Codebase**: `legacy/engine-v2/` (~7,943 lines of Rust)
**Edition**: Rust 2024
**Date**: 2026-02-15

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Build System & Dependencies](#3-build-system--dependencies)
4. [Module-by-Module Analysis](#4-module-by-module-analysis)
   - 4.1 [Entry Point — `main.rs`](#41-entry-point--mainrs)
   - 4.2 [Library Root — `lib.rs`](#42-library-root--librs)
   - 4.3 [Core Module](#43-core-module)
     - 4.3.1 [`core::types`](#431-coretypes)
     - 4.3.2 [`core::context`](#432-corecontext)
     - 4.3.3 [`core::kernels`](#433-corekernels)
     - 4.3.4 [`core::backend`](#434-corebackend)
   - 4.4 [Backends Module](#44-backends-module)
     - 4.4.1 [`backends::tensorrt`](#441-backendstensorrt)
   - 4.5 [Codecs Module](#45-codecs-module)
     - 4.5.1 [`codecs::sys`](#451-codecssys)
     - 4.5.2 [`codecs::nvdec`](#452-codecsnvdec)
     - 4.5.3 [`codecs::nvenc`](#453-codecsnvenc)
   - 4.6 [Engine Module](#46-engine-module)
     - 4.6.1 [`engine::pipeline`](#461-enginepipeline)
     - 4.6.2 [`engine::inference`](#462-engineinference)
   - 4.7 [I/O Module](#47-io-module)
     - 4.7.1 [`io::probe`](#471-ioprobe)
     - 4.7.2 [`io::ffmpeg_demuxer`](#472-ioffmpeg_demuxer)
     - 4.7.3 [`io::ffmpeg_muxer`](#473-ioffmpeg_muxer)
     - 4.7.4 [`io::ffmpeg_sys`](#474-ioffmpeg_sys)
     - 4.7.5 [`io::file_source`](#475-iofile_source)
     - 4.7.6 [`io::file_sink`](#476-iofile_sink)
   - 4.8 [Utilities](#48-utilities)
     - 4.8.1 [`error`](#481-error)
     - 4.8.2 [`debug_alloc`](#482-debug_alloc)
     - 4.8.3 [`probe_ort`](#483-probe_ort)
5. [CLI & Configuration Reference](#5-cli--configuration-reference)
6. [Data Flow & Inter-Module Interactions](#6-data-flow--inter-module-interactions)
7. [Architectural Patterns & Design Decisions](#7-architectural-patterns--design-decisions)
8. [Safety & Correctness Analysis](#8-safety--correctness-analysis)
9. [Potential Improvements & Risks](#9-potential-improvements--risks)

---

## 1. Executive Summary

RAVE v2.0 is a GPU-native video super-resolution engine written in Rust. It upscales video files using AI inference (ONNX models executed via TensorRT) while keeping **all frame data exclusively on the GPU** throughout the entire pipeline. The architecture is:

```
NVDEC (HW decode) → CUDA Preprocess → TensorRT Inference → CUDA Postprocess → NVENC (HW encode)
```

**Key design principles:**
- **Zero-copy GPU residency**: Frame data never touches host RAM during steady-state processing.
- **Bounded async pipeline**: Four concurrent stages connected by `tokio::sync::mpsc` channels with configurable backpressure.
- **Bucketed memory pool**: After warm-up, all device memory comes from a recycling pool — zero CUDA driver allocations in steady state.
- **RAII resource management**: All GPU resources (device memory, events, decoder/encoder handles) are cleaned up via Rust's ownership and `Drop` semantics.
- **Structured telemetry**: Every error variant maps to a stable numeric code; per-stage latency and VRAM usage are tracked via lock-free atomic counters.

The codebase is approximately 7,943 lines across 26 Rust source files, organized into 6 primary modules (`core`, `backends`, `codecs`, `engine`, `io`, `error`) plus utility modules.

---

## 2. Architecture Overview

### High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              RAVE v2.0 Pipeline                           │
│                                                                                  │
│  ┌──────────┐  ch(4)  ┌────────────┐  ch(2)  ┌───────────┐  ch(4)  ┌──────────┐│
│  │ Decoder  │────────►│ Preprocess │────────►│ Inference │────────►│ Encoder  ││
│  │ (NVDEC)  │         │ (CUDA kern)│         │(TensorRT) │         │ (NVENC)  ││
│  │ blocking │         │   async    │         │   async   │         │ blocking ││
│  └──────────┘         └────────────┘         └───────────┘         └──────────┘│
│       │                     │                      │                     │      │
│  NV12 (GPU)           RGB F32/F16            RGB F32/F16           NV12 (GPU)  │
│                        NCHW planar           NCHW planar                       │
│                                                                                  │
│  ═══════════════════════ All data GPU-resident ═══════════════════════════════  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
main.rs
  ├── core::context  (GpuContext, buffer pool, VRAM accounting)
  ├── core::kernels  (PreprocessKernels, NVRTC compilation)
  ├── core::backend  (UpscaleBackend trait, ModelMetadata)
  ├── core::types    (GpuTexture, FrameEnvelope, PixelFormat)
  ├── backends::tensorrt  (TensorRtBackend, OutputRing, ORT session)
  ├── codecs::nvdec  (NvDecoder, BitstreamSource, parser callbacks)
  ├── codecs::nvenc  (NvEncoder, BitstreamSink, registration cache)
  ├── codecs::sys    (raw FFI: nvcuvid, nvEncodeAPI, CUDA driver)
  ├── engine::pipeline  (UpscalePipeline, stage functions, metrics)
  ├── engine::inference (InferencePipeline — end-to-end helper)
  ├── io::probe      (FFmpeg container probing)
  ├── io::ffmpeg_demuxer  (BitstreamSource for containers)
  ├── io::ffmpeg_muxer    (BitstreamSink for containers)
  ├── io::file_source     (BitstreamSource for raw bitstreams)
  ├── io::file_sink       (BitstreamSink for raw bitstreams)
  └── error          (EngineError, error codes, Result alias)
```

---

## 3. Build System & Dependencies

### `Cargo.toml`

| Dependency | Version | Purpose |
|---|---|---|
| `clap` | 4 (derive) | CLI argument parsing |
| `cudarc` | 0.12 (driver, nvrtc, cuda-12060) | CUDA Driver API safe wrappers, NVRTC kernel compilation |
| `ort` | 2.0.0-rc.11 (cuda, tensorrt) | ONNX Runtime with TensorRT + CUDA execution providers |
| `tokio` | 1 (rt-multi-thread, sync, macros, signal, time) | Async runtime for pipeline orchestration |
| `tokio-util` | 0.7 (rt) | CancellationToken for graceful shutdown |
| `thiserror` | 2 | Library-grade typed error derive |
| `anyhow` | 1 | Application-level error wrapping |
| `async-trait` | 0.1 | Dynamic dispatch for `UpscaleBackend` trait |
| `tracing` | 0.1 | Structured logging |
| `tracing-subscriber` | 0.3 (env-filter) | Log output with `RUST_LOG` env filter |
| `half` | 2 (num-traits) | IEEE 754 half-precision (f16) tensor support |
| `ffmpeg-sys-next` | 8 | FFI bindings for FFmpeg container mux/demux |

### `build.rs`

The build script resolves three native dependency paths:

1. **CUDA Toolkit** (`CUDA_PATH` env var): Links `cuda.lib` (driver API). Searches `lib/x64` (Windows) or `lib64` (Linux).
2. **Video Codec SDK**: Links `nvcuvid.lib` (NVDEC) and `nvencodeapi.lib` (NVENC). Searches `../third_party/nvcodec` first, falls back to CUDA toolkit lib directory.
3. **FFmpeg** (`FFMPEG_DIR` env var): Adds `lib/` to the linker search path for `avcodec`, `avformat`, etc. Set in `.cargo/config.toml`.

### `.cargo/config.toml`

Sets three environment variables for the build:
- `FFMPEG_DIR` — Points to the bundled FFmpeg distribution.
- `LIBCLANG_PATH` — Required by `bindgen` (used internally by `ffmpeg-sys-next`) for C header parsing.
- `PKG_CONFIG = "nonexistent"` — Disables pkg-config to prevent conda/MSYS2 interference on Windows.

### Feature Flags

| Feature | Default | Purpose |
|---|---|---|
| `debug-alloc` | Off | Wraps global allocator to count host heap allocations during steady-state processing. Must NOT be enabled in release builds. |

---

## 4. Module-by-Module Analysis

### 4.1 Entry Point — `main.rs`

**File**: `src/main.rs` (373 lines)
**Role**: CLI entrypoint — parses arguments, validates hardware, constructs all pipeline components, and runs the async pipeline.

**Functionality:**

1. Initializes `tracing` with `RUST_LOG` env-filter (default: `info`).
2. Parses CLI arguments via `clap::Parser` (the `Cli` struct).
3. Builds a `tokio` multi-threaded runtime.
4. Calls `run_pipeline()` which orchestrates 11 sequential setup steps:
   - Parse precision (`fp32`/`fp16`)
   - Detect input/output type (container vs. raw bitstream)
   - Probe container metadata if applicable
   - Initialize `GpuContext` on the selected CUDA device
   - Set optional VRAM limit
   - Compile CUDA preprocessing kernels via NVRTC
   - Create and initialize the `TensorRtBackend` (ORT session, TRT engine)
   - Query model metadata (scale factor, tensor names)
   - Create decoder (`NvDecoder`) with appropriate `BitstreamSource`
   - Create encoder (`NvEncoder`) with appropriate `BitstreamSink`
   - Assemble and run `UpscalePipeline`
5. Reports pool stats and VRAM usage on completion.
6. Exits with error code 0 on success or the `EngineError::error_code()` on failure.

**Container detection**: `is_container()` checks file extensions against `CONTAINER_EXTENSIONS` (`mp4`, `mkv`, `mov`, `avi`, `webm`, `ts`, `flv`). Container inputs use `FfmpegDemuxer`; raw bitstreams use `FileBitstreamSource`. Same logic applies to output (container → `FfmpegMuxer`, raw → `FileBitstreamSink`).

**Notable design**: The raw CUcontext handle for NVENC is extracted via `transmute` from cudarc's `CudaDevice` (line 316-322). This is fragile but necessary because cudarc does not expose a public accessor for the underlying `CUcontext`.

**CLI flags used**: All flags defined in the `Cli` struct (see [Section 5](#5-cli--configuration-reference)).

---

### 4.2 Library Root — `lib.rs`

**File**: `src/lib.rs` (31 lines)
**Role**: Crate root — declares public modules and conditionally installs the debug allocator.

Re-exports: `backends`, `codecs`, `core`, `debug_alloc`, `engine`, `error`, `io`, `probe_ort`.

When `debug-alloc` feature is active, installs `TrackingAllocator` as the `#[global_allocator]`.

---

### 4.3 Core Module

#### 4.3.1 `core::types`

**File**: `src/core/types.rs` (220 lines)
**Role**: GPU-resident frame types and pixel format contracts. The foundational data types used by every other module.

**Key types:**

- **`PixelFormat`** (enum): Defines four on-device pixel formats:
  - `Nv12` — NVDEC native output (Y + UV semi-planar, 4:2:0)
  - `RgbPlanarF32` — Inference I/O (3 × H × W × 4 bytes, NCHW, [0,1])
  - `RgbPlanarF16` — Inference I/O for half-precision models
  - `RgbInterleavedU8` — For future NVENC RGB input support

  Each variant provides `byte_size()`, `element_bytes()`, and `channels()` methods for deterministic buffer sizing.

- **`GpuTexture`**: A single video frame entirely in GPU device memory. Fields:
  - `data: Arc<CudaSlice<u8>>` — RAII device memory with reference counting
  - `width`, `height` — Frame dimensions
  - `pitch` — Row stride in bytes (may exceed `width × bpp` for alignment)
  - `format` — Pixel format

  **Ownership model**: `Arc<CudaSlice<u8>>` provides:
  - RAII: memory freed when last reference drops
  - Send + Sync: safe to share across pipeline stages and tokio tasks
  - Cheap clone: incrementing `Arc` refcount, no device memory copy

  **Compile-time proof**: A `const` block with `assert_send_sync::<GpuTexture>()` statically verifies the Send + Sync bounds.

- **`FrameEnvelope`**: A `GpuTexture` annotated with pipeline metadata (`frame_index`, `pts`, `is_keyframe`). This is the unit that flows through bounded channels between pipeline stages.

**Design decision**: EOS (end-of-stream) is encoded as channel closure (sender drop), not as a sentinel value. The protocol is: Decoder closes tx → Preprocess sees `None` → closes its tx → ... → Encoder returns.

---

#### 4.3.2 `core::context`

**File**: `src/core/context.rs` (866 lines)
**Role**: Shared GPU context — the single most critical infrastructure module. Provides device management, stream management, bucketed buffer pool, VRAM accounting, performance profiling, and health monitoring.

**Key types:**

- **`GpuContext`**: Long-lived shared context (`Arc<GpuContext>`) used by every pipeline stage.
  - `device: Arc<CudaDevice>` — cudarc device handle
  - Three dedicated CUDA streams: `decode_stream`, `preprocess_stream`, `inference_stream` — enabling concurrent GPU execution on different hardware units
  - `buffer_pool: Mutex<BucketedPool>` — bucketed recycling pool
  - `pool_stats: PoolStats` — lock-free atomic hit/miss/recycle/overflow counters
  - `vram: VramAccounting` — atomic current/peak byte counters
  - `alloc_policy: AllocPolicy` — tracks warm-up vs. steady-state
  - `profiler: PerfProfiler` — per-stage GPU timing via atomic accumulators
  - `queue_depth: QueueDepthTracker` — per-stage queue depth for backpressure observability
  - `vram_limit: AtomicUsize` — advisory VRAM cap

- **`BucketedPool`**: The zero-free steady-state buffer pool.
  - **Bucket sizing**: Sizes < 2 MiB → rounded up to next power-of-two (min 4096). Sizes >= 2 MiB → rounded up to next 2 MiB boundary. This eliminates fragmentation from small size variations.
  - **LIFO stack per bucket**: `HashMap<usize, Vec<CudaSlice<u8>>>`. `take()` and `put()` are O(1).
  - **Overflow cap**: 32 buffers per bucket. Overflow frees to driver and decrements VRAM accounting.
  - **Drain on shutdown**: `GpuContext::drop()` calls `pool.drain()` to release all pooled memory.

- **`VramAccounting`**: Atomic `current` and `peak` byte counters. Lock-free reads via `Relaxed` ordering. `fetch_max` for peak tracking.

- **`AllocPolicy`**: `AtomicBool` tracking warm-up vs. steady-state. Pool misses in steady state trigger tracing warnings.

- **`PerfProfiler`**: Atomic accumulators for per-stage GPU timing (preprocess, inference, postprocess, encode) plus kernel launch overhead and peak frame latency.

- **`StreamOverlapTimer`**: Measures overlap between decode and inference streams using CUDA events (`cuEventElapsedTime`). Negative elapsed time = concurrent execution proven. Stores samples internally for statistical reporting.

- **`QueueDepthTracker`**: Lock-free atomic counters per stage (decode, preprocess, inference) for observability.

- **`HealthSnapshot`**: Immutable snapshot of all health metrics for telemetry export.

**Key methods:**
- `alloc(size)` — Tries pool first, falls back to `CudaDevice::alloc_zeros`. Increments VRAM accounting only on pool miss. Logs VRAM limit violations.
- `alloc_aligned(size, alignment)` — Rounds up to alignment boundary, then delegates to `alloc()`.
- `recycle(buf)` — Returns buffer to pool. Frees and decrements accounting on overflow.
- `prefetch_l2(device_ptr, count, stream)` — Async L2 cache prefetch hint via `cuMemPrefetchAsync`.
- `sync_stream(stream)` / `sync_all()` — Blocking stream synchronization.

**Safety**: `unsafe impl Send` and `unsafe impl Sync` for `GpuContext`. Justified by: CUDA driver API is thread-safe for distinct streams, and internal state is protected by `Mutex`/atomics.

**Tests**: Unit tests for bucket sizing edge cases (zero, small, large, boundary values).

---

#### 4.3.3 `core::kernels`

**File**: `src/core/kernels.rs` (873 lines)
**Role**: CUDA preprocessing kernels compiled via NVRTC. All color space conversions and precision transforms run entirely on-device.

**CUDA C Kernels (embedded as `const PREPROCESS_CUDA_SRC`):**

| Kernel | Input | Output | Use Case |
|---|---|---|---|
| `nv12_to_rgb_planar_f32` | NV12 (Y+UV) | RgbPlanarF32 NCHW [0,1] | F32 model preprocess |
| `nv12_to_rgb_planar_f16` | NV12 (Y+UV) | RgbPlanarF16 NCHW [0,1] | F16 model preprocess (fused, no F32 intermediate) |
| `f32_to_f16` | RgbPlanarF32 | RgbPlanarF16 | Element-wise truncation |
| `f16_to_f32` | RgbPlanarF16 | RgbPlanarF32 | Element-wise promotion |
| `rgb_planar_f32_to_nv12` | RgbPlanarF32 | NV12 | Postprocess for encoder |

**Color space**: BT.709 (HD/4K standard). Full-range conversion. Not configurable at runtime.

**Compilation**: NVRTC compiles CUDA C to PTX once at engine startup. Options: `ftz=true` (flush denorms), `prec_div=false` (fast division), `prec_sqrt=false`. The PTX is loaded into the device as a named module (`rave_preprocess`), and kernel function handles are resolved once and reused.

**Key types:**

- **`PreprocessKernels`**: Holds resolved `CudaFunction` handles for all 5 kernels. Created once, immutable thereafter.

- **`ModelInput`**: Annotates a `GpuTexture` with batch dimension metadata (`[1, C, H, W]`) without any data copy. Zero-copy batch injection.

- **`PreprocessPipeline`**: Encapsulates the full transform chain:
  - `prepare()`: NV12 → model-ready tensor (selects F32 or F16 path based on `ModelPrecision`)
  - `postprocess()`: model output → NV12 (handles F16→F32 promotion if needed)

- **`KernelTimer`**: GPU-side timing using `cuEventRecord` before/after kernel launch. Non-blocking measurement.

- **`StageMetrics`**: Accumulated per-stage latency (`total_ms`, `launch_count`, `avg_ms()`).

**Launch configs:**
- 2D kernels: 16×16 thread blocks, grid = ceil(width/16) × ceil(height/16)
- 1D kernels: 256 threads per block

**NV12→RGB kernel details**: The NV12 format has Y plane at full resolution and UV plane at half resolution (4:2:0 chroma subsampling). Each thread computes one pixel: reads Y at `(x, y)`, reads UV at `(x/2, y/2)`, applies BT.709 matrix, clamps to [0,1], writes to NCHW planar output.

**RGB→NV12 kernel details**: Inverse BT.709 conversion with chroma subsampling. Only even `(x, y)` threads write UV values, averaging the 2×2 block for proper downsampling.

---

#### 4.3.4 `core::backend`

**File**: `src/core/backend.rs` (105 lines)
**Role**: Defines the `UpscaleBackend` trait — the GPU-only inference contract that all backend implementations must satisfy.

**`UpscaleBackend` trait (async_trait, dyn-dispatch):**

| Method | Signature | Contract |
|---|---|---|
| `initialize()` | `async fn` → `Result<()>` | Load model, allocate GPU resources, build TRT engine. Called exactly once. |
| `process(input)` | `async fn(GpuTexture)` → `Result<GpuTexture>` | Run inference. Input must be `RgbPlanarF32` or `RgbPlanarF16`. Output is same format, spatially upscaled. |
| `shutdown()` | `async fn` → `Result<()>` | Release all GPU resources in defined order. |
| `metadata()` | `fn` → `Result<&ModelMetadata>` | Query model metadata (scale, tensor names, dimension constraints). |

**`ModelMetadata`**: Scale factor, input/output tensor names, channel count, min/max supported spatial dimensions.

**Four invariants**: (1) GPU-resident I/O only, (2) pre-allocated output buffers, (3) Send + Sync, (4) deterministic cleanup via `shutdown()` + `Drop` safety net.

---

### 4.4 Backends Module

#### 4.4.1 `backends::tensorrt`

**File**: `src/backends/tensorrt.rs` (941 lines)
**Role**: Concrete `UpscaleBackend` implementation using ONNX Runtime with TensorRT Execution Provider and IO Binding for zero-copy GPU inference.

**Key types:**

- **`TensorRtBackend`**: The primary inference engine.
  - `model_path`, `ctx: Arc<GpuContext>`, `device_id`
  - `ring_size`, `min_ring_slots` — output buffer ring configuration
  - `meta: OnceLock<ModelMetadata>` — lazily populated on `initialize()`
  - `state: Mutex<Option<InferenceState>>` — tokio Mutex for `(Session, OutputRing)`
  - `inference_metrics: InferenceMetrics` — atomic latency tracking
  - `precision_policy: PrecisionPolicy` — FP32 / FP16 / INT8
  - `batch_config: BatchConfig` — max batch size and latency deadline

- **`OutputRing`**: Fixed-size ring of pre-allocated `Arc<CudaSlice<u8>>` device buffers.
  - **Serialization invariant**: `acquire()` asserts `Arc::strong_count == 1` before returning a slot, guaranteeing the previous consumer has finished reading.
  - **Ring sizing**: Must be >= `downstream_channel_capacity + 2` to prevent deadlock (ring must have enough slots for the channel to drain before wrapping).
  - **Lazy initialization**: Ring is created on the first `process()` call (not in `initialize()`) because input dimensions are needed.
  - **Reallocation**: If input dimensions change, all slots are reallocated.
  - **Metrics**: `RingMetrics` tracks slot reuse, contention events, and first-use counts.

- **`PrecisionPolicy`** (enum): `Fp32`, `Fp16`, `Int8 { calibration_table }`. Controls TensorRT EP optimization flags.

- **`BatchConfig`**: `max_batch` and `latency_deadline_us` for future batched inference.

- **`InferenceMetrics`**: Atomic counters for frames inferred, cumulative and peak inference time.

**`initialize()` flow:**
1. Build TensorRT EP with device ID, engine cache, and precision flags.
2. Create ORT `Session` with TensorRT EP only — no CUDA EP, no CPU EP fallback.
3. Validate providers (structural guarantee: successful session creation = all nodes on TRT).
4. Extract `ModelMetadata` from ONNX tensor descriptors.
5. Store session in `InferenceState`.

**`process()` flow:**
1. Validate input format (accepts `RgbPlanarF32` or `RgbPlanarF16`).
2. Lazy-init or realloc the `OutputRing`.
3. Acquire next ring slot (asserts `strong_count == 1`).
4. Create ORT IO Binding with direct GPU device pointers (zero-copy via `create_tensor_from_device_memory`).
5. Run `session.run_binding()` — synchronous, blocks until all TensorRT kernels complete.
6. Record inference latency.
7. Optionally verify zero host allocations (debug-alloc feature).
8. Return `GpuTexture` wrapping the ring slot `Arc`.

**Zero-copy IO Binding**: Uses the raw ORT C API (`CreateTensorWithDataAsOrtValue`) to create ORT tensors directly from GPU device pointers. No host staging, no ORT-managed device allocation.

**`create_tensor_from_device_memory()`**: Unsafe helper that creates an `OrtValue` tensor from a raw `*mut c_void` device pointer, byte count, shape, and element type using the ORT C API (`CreateMemoryInfo` for CUDA, then `CreateTensorWithDataAsOrtValue`).

**CUDA stream ordering**: ORT creates its own internal CUDA stream for TensorRT EP execution. `run_with_binding()` is synchronous — it blocks until all GPU work completes. Therefore, CUDA global memory coherency guarantees visibility to subsequent readers on any stream.

**Pointer identity audit**: `verify_pointer_identity()` uses `debug_assert_eq!` to verify that IO-bound device pointers match the source `GpuTexture` and ring slot pointers exactly. This catches any ORT-internal buffer reallocation.

**Shutdown**: Syncs all streams, reports ring and inference metrics, then drops the session and ring in order.

---

### 4.5 Codecs Module

#### 4.5.1 `codecs::sys`

**File**: `src/codecs/sys.rs` (846 lines)
**Role**: Raw FFI bindings to NVIDIA Video Codec SDK (nvcuvid + nvEncodeAPI) and CUDA driver event/memcpy functions.

**Coverage:**

| API | Functions/Types |
|---|---|
| **NVDEC (nvcuvid)** | `cuvidCreateVideoParser`, `cuvidParseVideoData`, `cuvidDestroyVideoParser`, `cuvidCreateDecoder`, `cuvidDecodePicture`, `cuvidMapVideoFrame64`, `cuvidUnmapVideoFrame64`, `cuvidDestroyDecoder` |
| **NVENC (nvEncodeAPI)** | `NvEncodeAPICreateInstance` → function pointer table with: `nvEncOpenEncodeSessionEx`, `nvEncInitializeEncoder`, `nvEncCreateBitstreamBuffer`, `nvEncDestroyBitstreamBuffer`, `nvEncEncodePicture`, `nvEncLockBitstream`, `nvEncUnlockBitstream`, `nvEncMapInputResource`, `nvEncUnmapInputResource`, `nvEncRegisterResource`, `nvEncUnregisterResource`, `nvEncGetEncodePresetConfigEx`, `nvEncDestroyEncoder` |
| **CUDA Driver** | `cuEventCreate`, `cuEventDestroy_v2`, `cuEventRecord`, `cuStreamWaitEvent`, `cuStreamSynchronize`, `cuMemcpy2DAsync_v2` |

**Enums**: `cudaVideoCodec` (MPEG1-AV1), `cudaVideoSurfaceFormat`, `cudaVideoChromaFormat`, `cudaVideoDeinterlaceMode`, `NV_ENC_DEVICE_TYPE`, `NV_ENC_BUFFER_FORMAT`, `NV_ENC_PIC_TYPE`, `NV_ENC_PIC_STRUCT`, `NV_ENC_TUNING_INFO`.

**Well-known GUIDs**: `NV_ENC_CODEC_HEVC_GUID`, `NV_ENC_CODEC_H264_GUID`, `NV_ENC_PRESET_P7_GUID`, `NV_ENC_HEVC_PROFILE_MAIN_GUID`, `NV_ENC_HEVC_PROFILE_MAIN10_GUID`.

**Helpers**: `check_cu(result, context)` and `check_nvenc(status, context)` convert raw C error codes to `EngineError::Decode`/`EngineError::Encode` with context strings.

**Version**: Targets NVENC API v12.2 (`nvenc_struct_version` computes versioned struct tags).

---

#### 4.5.2 `codecs::nvdec`

**File**: `src/codecs/nvdec.rs` (632 lines)
**Role**: NVDEC hardware decoder producing GPU-resident NV12 `FrameEnvelope`s. Implements `FrameDecoder`.

**Architecture:**

```
BitstreamSource → cuvidParseVideoData → (callbacks) → cuvidDecodePicture
                                                              ↓
                                                    cuvidMapVideoFrame64
                                                              ↓
                                                   cuMemcpy2DAsync (D2D)
                                                              ↓
                                                    our CudaSlice buffer
                                                              ↓
                                                    cuvidUnmapVideoFrame64
                                                              ↓
                                                   cuEventRecord(decode_done)
                                                              ↓
                                                GpuTexture { NV12, pitch-aligned }
```

**Key types:**

- **`BitstreamSource`** (trait): `read_packet()` → `Result<Option<BitstreamPacket>>`. Implementations provide compressed NAL units.
- **`BitstreamPacket`**: `data: Vec<u8>`, `pts: i64`, `is_keyframe: bool`. Host memory (compressed data is ~10 KB/frame, acceptable).
- **`NvDecoder`**: Owns the NVDEC parser, decoder, bitstream source, event pool, and callback state.
- **`EventPool`**: Reusable LIFO pool of CUDA events with `CU_EVENT_DISABLE_TIMING` (lightweight ordering-only events). Avoids per-frame event creation overhead.
- **`CallbackState`**: Shared mutable state between parser callbacks and the main decoder. Contains decoder handle, parsed format, pending display queue, and codec type.
- **`DecodedFrame`**: `FrameEnvelope` + `decode_event` (CUDA event for cross-stream sync).

**Parser callbacks (unsafe extern "C"):**
- `sequence_callback`: Creates/recreates the hardware decoder when a sequence header is parsed. Determines required decode surfaces (min 8).
- `decode_callback`: Enqueues a picture for decoding on the NVDEC hardware.
- `display_callback`: Pushes reordered display-ready pictures to `pending_display` queue.

**`decode_next()` flow:**
1. Check `pending_display` queue for decoded frames.
2. If available: `map_and_copy()` → D2D copy on `decode_stream` → record event → unmap NVDEC surface → return `FrameEnvelope`.
3. If empty: feed more bitstream packets to parser. If source exhausted, send EOS.
4. Loop until frame available or source exhausted.

**Why D2D copy?** NVDEC surfaces are a finite pool (8-16). They must be returned quickly via `cuvidUnmapVideoFrame64`. Copying to our buffer (~24 us for 4K NV12 at 500 GB/s) decouples decoder surface lifetime from pipeline frame lifetime.

**Cross-stream sync**: After D2D copy, `cuEventRecord(event, decode_stream)` is called. The preprocess stage must call `cuStreamWaitEvent(preprocess_stream, event, 0)` before reading. `wait_for_event()` is provided as a public helper.

**`get_raw_stream()`**: Extracts the raw `CUstream` handle from cudarc's `CudaStream` via pointer cast. **Fragile** — relies on cudarc 0.12's internal layout. Marked with a TODO to replace with `CudaStream::as_raw()` when available.

**Drop order**: Parser is destroyed first (stops callbacks), then decoder, then events are returned to pool.

---

#### 4.5.3 `codecs::nvenc`

**File**: `src/codecs/nvenc.rs` (514 lines)
**Role**: NVENC hardware encoder consuming GPU-resident NV12 `FrameEnvelope`s. Implements `FrameEncoder`.

**Architecture:**

```
GpuTexture { NV12 } → nvEncRegisterResource(CUDA)
                              ↓
                    nvEncMapInputResource
                              ↓
                     nvEncEncodePicture
                              ↓
                    nvEncLockBitstream → BitstreamSink.write_packet()
                              ↓
                    nvEncUnlockBitstream
                    nvEncUnmapInputResource
```

**Key types:**

- **`BitstreamSink`** (trait): `write_packet(data, pts, is_keyframe)` and `flush()`.
- **`NvEncConfig`**: Encoder parameters — width, height, fps, bitrate, max_bitrate, GOP length, B-frames, NV12 pitch.
- **`RegistrationCache`**: `HashMap<u64, *mut c_void>` caching NVENC resource registrations keyed by device pointer. Avoids per-frame `nvEncRegisterResource` overhead.
- **`NvEncoder`**: Owns encoder session, function pointer table, bitstream buffer, sink, registration cache, and config.

**Initialization:**
1. `NvEncodeAPICreateInstance` → function pointer table.
2. `nvEncOpenEncodeSessionEx` with CUDA device type.
3. `nvEncGetEncodePresetConfigEx` → P7 High Quality preset for HEVC.
4. Configure: HEVC Main profile, GOP length, B-frames, VBR or CQP rate control.
5. `nvEncInitializeEncoder` with configured params.
6. `nvEncCreateBitstreamBuffer` for output.

**Encode flow (per frame):**
1. Get or create resource registration for the device pointer.
2. `nvEncMapInputResource` — makes the device pointer accessible to NVENC.
3. `nvEncEncodePicture` — submits the frame for encoding.
4. `nvEncLockBitstream` — blocks until encoded data is ready.
5. Copy encoded data to `BitstreamSink` via `write_packet()`.
6. `nvEncUnlockBitstream`, `nvEncUnmapInputResource`.

**IDR forcing**: If `frame.is_keyframe`, sets `NV_ENC_PIC_FLAG_FORCEIDR` on the encode params.

**EOS**: `nvEncEncodePicture` with `NV_ENC_PIC_FLAG_EOS` flushes the encoder.

**Drop**: Unregisters all cached resources, destroys bitstream buffer, destroys encoder session.

---

### 4.6 Engine Module

#### 4.6.1 `engine::pipeline`

**File**: `src/engine/pipeline.rs` (1,185 lines)
**Role**: The core pipeline orchestrator. Connects four concurrent stages via bounded channels with backpressure, cancellation, metrics, and shutdown coordination.

**Key types:**

- **`FrameDecoder`** (trait): `decode_next()` → `Result<Option<FrameEnvelope>>`. Implemented by `NvDecoder` and `MockDecoder`.
- **`FrameEncoder`** (trait): `encode(frame)` and `flush()`. Implemented by `NvEncoder` and `MockEncoder`.
- **`PipelineConfig`**: Channel capacities (`decoded_capacity`, `preprocessed_capacity`, `upscaled_capacity`), `encoder_nv12_pitch`, `model_precision`, `enable_profiler`.
- **`PipelineMetrics`**: Atomic per-stage frame counters and latency accumulators.
- **`UpscalePipeline`**: Owns `GpuContext`, `PreprocessKernels`, `PipelineConfig`, `CancellationToken`, and `PipelineMetrics`.

**`run()` — Pipeline execution:**

1. Creates three `mpsc::channel` instances with configured capacities.
2. Spawns four tasks on a `JoinSet`:
   - **Stage 1 (Decode)**: `spawn_blocking` — calls `decoder.decode_next()` in a loop, sends `FrameEnvelope` via `blocking_send`. NVDEC may DMA-block, so this must be on a blocking thread.
   - **Stage 2 (Preprocess)**: `spawn` (async) — receives NV12 frames, runs `PreprocessPipeline::prepare()` (NV12→RGB via CUDA kernel), syncs `preprocess_stream`, recycles consumed NV12 buffer, sends preprocessed `FrameEnvelope`.
   - **Stage 3 (Inference + Postprocess)**: `spawn` (async) — receives preprocessed frames, runs `backend.process()` (TensorRT inference), then `PreprocessPipeline::postprocess()` (RGB→NV12), syncs `inference_stream`, recycles consumed RGB buffer, sends upscaled NV12 `FrameEnvelope`.
   - **Stage 4 (Encode)**: `spawn_blocking` — calls `blocking_recv` (pull model — encode pace drives throughput), calls `encoder.encode()`, always calls `flush()` before returning.
3. Collects results from `JoinSet`. On any error, cancels remaining stages.
4. Post-shutdown: syncs all CUDA streams, validates metric ordering invariants (`decoded >= preprocessed >= inferred >= encoded`), reports final metrics.

**Backpressure**: All channels are bounded. When downstream can't keep up, upstream `send().await` or `blocking_send()` suspends. No dropped frames, no spin loops.

**Shutdown protocol:**
- **Normal EOS**: Decoder exhausts input → drops tx → cascade through all stages.
- **Cancellation**: `CancellationToken::cancel()` → every stage checks `is_cancelled()` → drops sender → cascade.
- **Error**: Stage returns `Err` → sender drops → cascade. First error is propagated.

**Buffer recycling**: Both preprocess and inference stages attempt `Arc::try_unwrap()` on consumed frame data. If successful (no other references), the buffer is returned to `GpuContext::recycle()` for pool reuse.

**`stress_test_synthetic()`**: Two-phase stress test:
1. **Warm-up (5s)**: Populates the buffer pool with `MockDecoder`/`MockEncoder`.
2. **Measured run**: Tracks frame counts, latencies, VRAM stability, pool hit rate. Validates: VRAM stays within stable envelope, `decoded == encoded`, pool hit rate >= 90%.

**`AuditSuite`**: Phase 7 invariant checker. Runs a synthetic pipeline and validates:
1. **Residency**: Zero host allocations during hot path (via `debug_alloc`).
2. **Determinism**: VRAM delta <= 2 MiB between start and end (no leaks).
3. **Pool stability**: Hit rate >= 90% after warm-up.
4. **Concurrency**: `decoded == encoded` with no stalls.

**Mock types**: `MockDecoder` emits zeroed NV12 frames at ~60 FPS cadence (16.667ms sleep). `MockEncoder` counts frames and drops them.

---

#### 4.6.2 `engine::inference`

**File**: `src/engine/inference.rs` (164 lines)
**Role**: End-to-end GPU-resident inference pipeline helper. Composes preprocess + inference + postprocess into a single `process_frame()` method.

**`InferencePipeline`:**
- Wraps `PreprocessPipeline`, `TensorRtBackend`, `GpuContext`, and per-stage `StageMetrics`.
- `process_frame(envelope, stream)`: NV12 → preprocess → inference → postprocess → NV12.
- `report_metrics()`: Logs preprocess/inference/postprocess averages and VRAM usage.

This is a convenience wrapper for simpler single-threaded usage. The main pipeline in `engine::pipeline` performs the same steps but distributed across async stages with channels.

---

### 4.7 I/O Module

#### 4.7.1 `io::probe`

**File**: `src/io/probe.rs` (153 lines)
**Role**: FFmpeg-based container metadata probing. Opens a container, finds the best video stream, extracts codec/resolution/framerate/duration.

**`probe_container(path)`** → `Result<ContainerMetadata>`:
1. `avformat_open_input` → `avformat_find_stream_info` → `av_find_best_stream(VIDEO)`.
2. Maps `AVCodecID` to `cudaVideoCodec`: H264, HEVC, VP9, AV1 supported.
3. Extracts `avg_frame_rate` (prefers over `r_frame_rate`).
4. Returns `ContainerMetadata` with codec, width, height, fps_num/den, time_base, duration_us.

**`FormatGuard`**: RAII wrapper for `AVFormatContext` ensuring `avformat_close_input` on all exit paths.

**CLI mapping**: Auto-detects `--codec`, `--width`, `--height`, `--fps-num`, `--fps-den` from the container. CLI overrides take precedence.

---

#### 4.7.2 `io::ffmpeg_demuxer`

**File**: `src/io/ffmpeg_demuxer.rs` (307 lines)
**Role**: Implements `BitstreamSource` for container files (MP4/MKV/MOV). Reads compressed video packets and converts from MP4 length-prefixed to Annex B NAL units.

**`FfmpegDemuxer`:**
1. `avformat_open_input` → `avformat_find_stream_info` → `av_find_best_stream`.
2. Initializes bitstream filter: `h264_mp4toannexb` or `hevc_mp4toannexb`. VP9/AV1 pass through without BSF.
3. `read_packet()` loop: read from container → skip non-video → send to BSF → receive filtered Annex B packet.
4. PTS rescaling from stream time_base to microseconds via `av_rescale_q`.
5. EOS: flushes BSF to drain remaining packets.

**Drop**: Frees packets, BSF context, format context in reverse allocation order.

---

#### 4.7.3 `io::ffmpeg_muxer`

**File**: `src/io/ffmpeg_muxer.rs` (229 lines)
**Role**: Implements `BitstreamSink` for container files. Writes encoded HEVC bitstream into a container.

**`FfmpegMuxer`:**
1. `avformat_alloc_output_context2` (auto-detects format from extension).
2. `avformat_new_stream` with HEVC codec params.
3. `avio_open` for file I/O.
4. Lazy `avformat_write_header` on first packet.
5. `write_packet()`: rescales PTS from microseconds to stream time_base, copies data to `AVPacket`, calls `av_interleaved_write_frame`.
6. `flush()`: writes container trailer via `av_write_trailer`.

---

#### 4.7.4 `io::ffmpeg_sys`

**File**: `src/io/ffmpeg_sys.rs` (76 lines)
**Role**: FFmpeg FFI helpers and BSF declarations missing from `ffmpeg-sys-next` v8.

Provides:
- `AVBSFContext` struct layout (manually declared since `ffmpeg-sys-next` v8 doesn't generate BSF bindings).
- `av_bsf_get_by_name`, `av_bsf_alloc`, `av_bsf_init`, `av_bsf_send_packet`, `av_bsf_receive_packet`, `av_bsf_free` FFI declarations.
- `check_ffmpeg(ret, context)`: Translates FFmpeg error codes to `EngineError::Demux` with `av_strerror` messages.
- `to_cstring(s)`: Safe `CString` conversion with NUL byte error handling.

---

#### 4.7.5 `io::file_source`

**File**: `src/io/file_source.rs` (73 lines)
**Role**: Implements `BitstreamSource` for raw Annex B bitstream files (`.264`, `.265`, `.hevc`).

Loads the entire file into memory on first `read_packet()` call. Returns it as a single large packet. Subsequent calls return `None` (EOS). NVDEC's parser handles NAL boundary detection internally.

---

#### 4.7.6 `io::file_sink`

**File**: `src/io/file_sink.rs` (85 lines)
**Role**: Implements `BitstreamSink` for raw HEVC Annex B bitstream output.

Uses a 4 MiB `BufWriter` for efficient I/O. Writes raw NAL units directly. Progress logging every 100 packets.

---

### 4.8 Utilities

#### 4.8.1 `error`

**File**: `src/error.rs` (147 lines)
**Role**: Typed error hierarchy using `thiserror`.

**`EngineError` variants and codes:**

| Code Range | Category | Variants |
|---|---|---|
| 1xx | CUDA/driver | `Cuda(DriverError)` (100), `NvrtcCompile(CompileError)` (101) |
| 2xx | Inference | `Inference(ort::Error)` (200), `ModelMetadata(String)` (201), `NotInitialized` (202) |
| 3xx | Codecs | `Decode` (300), `Encode` (301), `Demux` (302), `Mux` (303), `BitstreamFilter` (304), `Probe` (305) |
| 4xx | Pipeline | `ChannelClosed` (400), `Shutdown` (401), `Pipeline` (402) |
| 5xx | Type contracts | `FormatMismatch` (500), `DimensionMismatch` (501), `BufferTooSmall` (502) |
| 6xx | Audit | `InvariantViolation` (600) |
| 7xx | Production | `PanicRecovered` (700), `VramLimitExceeded` (701), `BackpressureTimeout` (702), `DropOrderViolation` (703) |

**`is_recoverable()`**: Only `BackpressureTimeout` and `PanicRecovered` are considered recoverable.

**`error_code()`**: Stable integer for structured telemetry — enables error classification without string parsing.

---

#### 4.8.2 `debug_alloc`

**File**: `src/debug_alloc.rs` (84 lines)
**Role**: Debug-only host allocation tracker. Feature-gated (`--features debug-alloc`).

Wraps `std::alloc::System` in a `TrackingAllocator` that counts allocations via `AtomicUsize`. API: `enable()`, `disable()`, `reset()`, `count()`. When the feature is disabled, all functions are zero-cost no-ops.

Used by the TensorRT backend's `process()` method and the `AuditSuite` to verify zero host allocations during inference.

---

#### 4.8.3 `probe_ort`

**File**: `src/probe_ort.rs` (7 lines)
**Role**: ORT API exploration stub. Used during development to verify IO Binding types. Marked for removal before release.

---

## 5. CLI & Configuration Reference

```
rave -i <INPUT> -o <OUTPUT> -m <MODEL> [OPTIONS]
```

| Flag | Long | Type | Default | Module(s) | Description |
|---|---|---|---|---|---|
| `-i` | `--input` | `PathBuf` | required | main, io | Input video file (container or raw bitstream) |
| `-o` | `--output` | `PathBuf` | required | main, io | Output video file (container or raw bitstream) |
| `-m` | `--model` | `PathBuf` | required | backends::tensorrt | ONNX model path for super-resolution |
| `-p` | `--precision` | `String` | `"fp32"` | core::kernels, engine::pipeline | Model precision: `fp32`, `fp16` (aliases: `f32`, `f16`, `float32`, `float16`, `half`) |
| `-d` | `--device` | `usize` | `0` | core::context | CUDA device ordinal (0-indexed) |
| | `--vram-limit` | `usize` | `0` | core::context | VRAM limit in MiB (0 = unlimited). Advisory, not hard cap. |
| | `--decode-cap` | `usize` | `4` | engine::pipeline | Channel capacity: decode → preprocess |
| | `--preprocess-cap` | `usize` | `2` | engine::pipeline | Channel capacity: preprocess → inference |
| | `--upscale-cap` | `usize` | `4` | engine::pipeline, backends::tensorrt | Channel capacity: inference → encode. Also influences ring size. |
| | `--bitrate` | `u32` | `0` | codecs::nvenc | Output encoder bitrate in kbps (0 = CQP mode) |
| | `--codec` | `String?` | auto | main, codecs | Input codec: `h264`, `hevc` (auto-detected for containers) |
| | `--fps-num` | `u32?` | auto | main, codecs::nvenc, io | Framerate numerator (auto-detected for containers) |
| | `--fps-den` | `u32?` | auto | main, codecs::nvenc, io | Framerate denominator |
| | `--width` | `u32?` | auto | main, codecs | Input width (auto-detected for containers; default 1920 for raw) |
| | `--height` | `u32?` | auto | main, codecs | Input height (auto-detected for containers; default 1080 for raw) |

**Environment variables:**
| Variable | Purpose |
|---|---|
| `RUST_LOG` | Tracing filter (e.g., `debug`, `rave=trace`, `info`) |
| `CUDA_PATH` | CUDA toolkit root (set by NVIDIA installer) |
| `FFMPEG_DIR` | FFmpeg root with `lib/`, `include/`, `bin/` (set in `.cargo/config.toml`) |

---

## 6. Data Flow & Inter-Module Interactions

### Complete Frame Lifecycle

```
1. INPUT FILE
   │
   ├─ Container (.mp4/.mkv/.mov)
   │    io::probe::probe_container()    → ContainerMetadata (codec, w, h, fps)
   │    io::ffmpeg_demuxer::FfmpegDemuxer
   │       avformat_open_input → av_read_frame → BSF (mp4→annex_b) → BitstreamPacket
   │
   └─ Raw bitstream (.264/.265/.hevc)
        io::file_source::FileBitstreamSource
           fs::read → single BitstreamPacket

2. DECODE (blocking thread, decode_stream)
   codecs::nvdec::NvDecoder
     BitstreamPacket → cuvidParseVideoData → callbacks → cuvidDecodePicture
     cuvidMapVideoFrame64 → cuMemcpy2DAsync(D2D) → cuvidUnmapVideoFrame64
     cuEventRecord(decode_event, decode_stream)
     → FrameEnvelope { NV12 GpuTexture, frame_index, pts }

   [mpsc channel, capacity = decode-cap (default 4)]

3. PREPROCESS (async task, preprocess_stream)
   core::kernels::PreprocessPipeline::prepare()
     F32: nv12_to_rgb_planar_f32 kernel → RgbPlanarF32
     F16: nv12_to_rgb_planar_f16 kernel → RgbPlanarF16 (fused, no intermediate)
   ModelInput::from_texture() → batch dim annotation (zero copy)
   sync(preprocess_stream)
   recycle(NV12 buffer → pool)
   → FrameEnvelope { RgbPlanar[F32|F16] GpuTexture }

   [mpsc channel, capacity = preprocess-cap (default 2)]

4. INFERENCE + POSTPROCESS (async task, inference_stream)
   backends::tensorrt::TensorRtBackend::process()
     OutputRing::acquire() → pre-allocated device buffer
     ORT IO Binding (zero-copy: raw device ptrs → OrtValue)
     session.run_binding() → TensorRT kernels (synchronous)
     → upscaled RgbPlanar[F32|F16] GpuTexture (scale × dimensions)

   core::kernels::PreprocessPipeline::postprocess()
     F16→F32 promotion if needed (f16_to_f32 kernel)
     rgb_planar_f32_to_nv12 kernel → NV12 at encoder pitch
   sync(inference_stream)
   recycle(RGB buffer → pool)
   → FrameEnvelope { NV12 GpuTexture (upscaled) }

   [mpsc channel, capacity = upscale-cap (default 4)]

5. ENCODE (blocking thread)
   codecs::nvenc::NvEncoder::encode()
     nvEncRegisterResource (cached) → nvEncMapInputResource
     nvEncEncodePicture → nvEncLockBitstream → encoded data
     BitstreamSink::write_packet(data, pts, is_keyframe)
     nvEncUnlockBitstream → nvEncUnmapInputResource

6. OUTPUT FILE
   ├─ Container (.mp4/.mkv/.mov)
   │    io::ffmpeg_muxer::FfmpegMuxer
   │       avformat_write_header → av_interleaved_write_frame → av_write_trailer
   └─ Raw bitstream (.265/.hevc)
        io::file_sink::FileBitstreamSink
           BufWriter::write_all
```

### GPU Memory Flow

```
                   Pool alloc          Pool alloc          Ring slot
                      ↓                    ↓                   ↓
NV12 decode buf → [preprocess] → RGB tensor → [inference] → RGB output → [postprocess] → NV12 encode buf
     ↓                                  ↓                                       ↓
  recycle()                          recycle()                              (encode reads,
  (back to pool)                    (back to pool)                          then ring wraps)
```

All buffers flow through `GpuContext::alloc()` → use → `GpuContext::recycle()`. The `OutputRing` in the TensorRT backend pre-allocates a fixed set of buffers and rotates through them.

### Stream Concurrency

```
decode_stream:     [D2D copy frame N] [D2D copy frame N+1] ...
preprocess_stream: ─── wait(event) ── [NV12→RGB kernel N] ── [NV12→RGB kernel N+1] ...
inference_stream:  ──────────────────── [TRT inference N] ─── [postprocess N] ...
(ORT internal):    ──────────────────── [TRT kernels N] ───── ...
```

Cross-stream ordering is maintained via CUDA events (`cuEventRecord` on source stream, `cuStreamWaitEvent` on target stream).

---

## 7. Architectural Patterns & Design Decisions

### 7.1 Zero-Copy GPU Residency

The single most important architectural constraint: **frame pixel data never touches host RAM** during steady-state processing. This is enforced at multiple levels:

- `GpuTexture.data` is always `Arc<CudaSlice<u8>>` — device-resident memory.
- The `UpscaleBackend` trait contract requires GPU-resident I/O.
- ORT IO Binding uses raw device pointers via the C API (no ORT-managed host staging).
- NVENC input is registered directly from CUDA device pointers.
- The only host data is compressed bitstream packets (~10 KB/frame) — acceptable since these are codec-compressed.

**Exception**: NVENC `nvEncLockBitstream` copies encoded bitstream to host for `BitstreamSink::write_packet()`. This is unavoidable — encoded data must be written to disk.

### 7.2 Bucketed Memory Pool (Zero-Free Steady State)

After warm-up, the pool holds enough buffers for every allocation size. Subsequent `alloc()` calls are pool hits (O(1) LIFO pop), and `recycle()` returns buffers for reuse. No `cuMemFree` during normal operation.

**Bucketing strategy**:
- Sizes < 2 MiB: power-of-two (min 4096) — eliminates fragmentation from small kernel temporaries.
- Sizes >= 2 MiB: 2 MiB aligned — absorbs padding variations in frame buffers/tensors.
- Max 32 buffers per bucket — overflow freed to driver.

**VRAM accounting**: Atomic counters track current and peak usage. Incremented only on pool miss (fresh driver allocation). Not decremented on recycle (bytes remain allocated on device). Decremented only on overflow or drain.

### 7.3 RAII Resource Management

Every GPU resource has a corresponding `Drop` implementation:

| Resource | Owner | Drop behavior |
|---|---|---|
| Device memory (`CudaSlice`) | `Arc<CudaSlice<u8>>` in `GpuTexture` | `cuMemFree` via cudarc |
| Buffer pool | `GpuContext::drop()` | `pool.drain()` frees all pooled buffers |
| CUDA events | `EventPool::drop()`, `KernelTimer::drop()`, `StreamOverlapTimer::drop()` | `cuEventDestroy_v2` |
| NVDEC parser + decoder | `NvDecoder::drop()` | `cuvidDestroyVideoParser`, `cuvidDestroyDecoder` |
| NVENC session | `NvEncoder::drop()` | Unregister resources, destroy bitstream buffer, destroy encoder |
| ORT session | `TensorRtBackend::drop()` | `try_lock` → sync → drop session |
| FFmpeg contexts | `FfmpegDemuxer::drop()`, `FfmpegMuxer::drop()` | Free packets, BSF, format context |

### 7.4 Async Pipeline with Bounded Backpressure

Four stages on a tokio `JoinSet`:
- Decode and Encode use `spawn_blocking` (NVDEC/NVENC may DMA-block).
- Preprocess and Inference use `spawn` (async kernel launch + channel I/O).
- All channels are bounded → natural backpressure propagation.
- `CancellationToken` + `select!` (biased) provides cooperative cancellation.
- Encoder is the pull-model consumer — its `blocking_recv` pace drives overall throughput.

### 7.5 Output Ring Buffer Serialization

The `OutputRing` pre-allocates N device buffers wrapped in `Arc`. Each `acquire()` call checks `Arc::strong_count == 1` before returning, ensuring no concurrent reader. This provides:
- **Double-buffering guarantee**: Always a free slot for writing.
- **No allocation during inference**: Ring populated once, reused forever.
- **Contention detection**: Non-1 strong count triggers a contention event counter.
- **Ring sizing rule**: `ring_size >= downstream_channel_capacity + 2` prevents deadlock.

### 7.6 Cross-Stream Synchronization via CUDA Events

Instead of CPU-blocking `cuStreamSynchronize`, the pipeline uses non-blocking event-based ordering:
1. Decode records event on `decode_stream`.
2. Preprocess calls `cuStreamWaitEvent(preprocess_stream, event)`.
3. GPU handles ordering — CPU thread continues immediately.

This enables true pipeline parallelism where decode, preprocess, and inference can overlap on different hardware units (NVDEC ASIC, CUDA cores, TensorRT/Tensor Cores, NVENC ASIC).

### 7.7 TensorRT Engine Caching

The TensorRT EP is configured with `with_engine_cache(true)` and a `trt_cache/` directory next to the model. On first run, TRT builds an optimized engine (can take minutes). On subsequent runs, the cached engine is loaded instantly.

### 7.8 Structured Error Hierarchy

Every error variant maps to a stable numeric code (1xx-7xx). `is_recoverable()` distinguishes transient from fatal errors. The process exit code is the error code.

---

## 8. Safety & Correctness Analysis

### 8.1 Unsafe Code Audit

The codebase contains significant `unsafe` usage, all concentrated in FFI boundaries:

| Module | Unsafe Pattern | Risk Level | Justification |
|---|---|---|---|
| `codecs::nvdec` | Parser callbacks, `cuvidMapVideoFrame64`, `cuMemcpy2DAsync`, `cuEventRecord` | Medium | All NVIDIA-documented API patterns. Callback state lifetime is guaranteed by `Box<CallbackState>` outliving the parser. |
| `codecs::nvdec::get_raw_stream()` | `transmute`-like pointer cast to extract `CUstream` from cudarc | **High** | Relies on cudarc 0.12 internal layout. Will break on cudarc version change. |
| `codecs::nvenc` | NVENC function pointer calls, `nvEncLockBitstream` → `slice::from_raw_parts` | Medium | Standard NVENC API usage. Bitstream size is reported by NVENC. |
| `codecs::sys` | All extern "C" declarations | Low | Pure FFI declarations matching SDK headers. |
| `backends::tensorrt` | `create_tensor_from_device_memory` via ORT C API | Medium | Creates ORT tensors from raw device pointers. Pointer validity depends on caller. |
| `main.rs` | CUcontext extraction via pointer cast | **High** | Same fragility as `get_raw_stream()`. Depends on cudarc internal layout. |
| `core::context` | `cuMemPrefetchAsync` extern declaration | Low | Performance hint, non-fatal on failure. |
| `core::kernels` | `launch_on_stream` CUDA kernel launches | Medium | Bounds are verified by launch config matching frame dimensions. |
| `io::ffmpeg_*` | All FFmpeg API calls | Medium | Standard FFmpeg API patterns. RAII guards ensure cleanup. |
| `debug_alloc` | `GlobalAlloc` impl | Low | Delegates to `System` allocator. |

### 8.2 Send/Sync Safety

- `GpuContext`: `unsafe impl Send + Sync` — justified by Mutex/atomic internal synchronization. CUDA Driver API is thread-safe for distinct streams.
- `NvDecoder`: `unsafe impl Send` — used only from the decode blocking task (single thread).
- `NvEncoder`: `unsafe impl Send` — used only from the encode blocking task (single thread).
- `FfmpegDemuxer`/`FfmpegMuxer`: `unsafe impl Send` — all FFmpeg operations on a single thread.
- `GpuTexture`: Compile-time `assert_send_sync` proof via const evaluation.

### 8.3 Potential Soundness Concerns

1. **`get_raw_stream()` and CUcontext extraction**: These rely on cudarc's internal struct layout via pointer casts. A cudarc version bump could silently produce incorrect values, leading to CUDA API misuse. **Mitigation**: Pin cudarc version tightly; add runtime validation if possible.

2. **Parser callback state**: `CallbackState` is accessed via raw pointer from C callbacks. Lifetime is guaranteed by `Box` allocation, but there's no compile-time proof that callbacks don't outlive the `NvDecoder`.

3. **`OutputRing` strong_count check**: `Arc::strong_count` is advisory and not atomic with respect to the subsequent access. In theory, a race could occur if another thread drops its `Arc` between the check and the use. In practice, this is safe because the pipeline is single-writer (only the inference stage calls `acquire`).

4. **ORT C API tensor creation**: `create_tensor_from_device_memory` passes raw device pointers to ORT. If the underlying `CudaSlice` is freed before ORT finishes using it, use-after-free occurs. Currently safe because `run_binding()` is synchronous.

---

## 9. Potential Improvements & Risks

### 9.1 High Priority

1. **Replace `get_raw_stream()` pointer cast**: Petition cudarc upstream for `CudaStream::as_raw()` or use a more robust extraction method. This is the single biggest fragility point.

2. **Replace CUcontext extraction in `main.rs`**: Same concern as above. The `transmute` of `CudaDevice` to get the raw `CUcontext` is extremely fragile.

3. **H.264 output support**: Currently NVENC is hardcoded to HEVC. Adding H.264 output would require parameterizing the codec GUID, profile GUID, and BSF in the muxer.

4. **Dynamic resolution support**: The `OutputRing` supports reallocation, but the pipeline doesn't handle resolution changes mid-stream (e.g., adaptive bitrate sources). The preprocess stage recompiles kernels per instance, which could be optimized.

5. **Cross-stream event usage in pipeline**: The `decode_event` from `NvDecoder` is not currently used by the preprocess stage in the main pipeline. The preprocess stage calls `sync_stream(preprocess_stream)` instead of using event-based ordering. This means decode and preprocess cannot overlap on the GPU. Using `wait_for_event` instead would enable true pipeline parallelism.

### 9.2 Medium Priority

6. **Batch inference**: `BatchConfig` is defined but not used. The pipeline processes one frame at a time. Implementing batched inference would significantly improve throughput on models with dynamic batch axes.

7. **Error recovery**: Currently any stage error cancels the entire pipeline. For transient errors (e.g., VRAM pressure), the pipeline could skip frames or retry.

8. **INT8 calibration table path**: The `PrecisionPolicy::Int8` variant stores the calibration table path but the `.with_int8_calibration_table()` call is commented out in `initialize()`.

9. **`FileBitstreamSource` loads entire file into memory**: For large raw bitstream files (multi-GB), this is problematic. A chunked reader would be more memory-efficient.

10. **Container output DTS handling**: The muxer sets `dts = pts` which is incorrect for B-frame reordering. This may cause playback issues with B-frame-enabled encodes.

11. **Progress reporting**: No user-facing progress indicator (percentage, ETA, FPS counter). The tracing logs provide debug info but not user-friendly progress.

### 9.3 Low Priority

12. **AV1 encode support**: NVENC supports AV1 on Ada Lovelace+ GPUs. Adding this would require new GUIDs and profile config.

13. **Multi-GPU support**: `GpuContext` is single-device. Extending to multi-GPU would require device-to-device transfers or partitioned frame processing.

14. **Color space configurability**: BT.709 is hardcoded in the CUDA kernels. BT.601 (SD content) and BT.2020 (HDR) would require runtime kernel selection or recompilation.

15. **10-bit pipeline**: Currently NV12 (8-bit) only. P016 (10-bit) support would require new kernel variants and NVENC format changes.

16. **`probe_ort` module**: Development stub that should be removed before release.

17. **NVENC preset configurability**: Hardcoded to P7 High Quality. Exposing preset selection via CLI would allow quality/speed tradeoffs.

### 9.4 Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| cudarc layout change breaks `get_raw_stream()` | Critical | Medium (on version bump) | Pin cudarc version; add compile-time assertions; upstream PR |
| ORT RC version instability | High | Medium | Pin exact ORT version; test before upgrading |
| VRAM exhaustion on large resolutions | High | Low (pool + limit) | Advisory VRAM limit; could add hard limit with allocation failure |
| BSF API missing from ffmpeg-sys-next | Medium | Low (stable API) | Manual FFI declarations in `ffmpeg_sys.rs` |
| NVENC resource registration leak | Medium | Low | `RegistrationCache` + `Drop` cleanup |
| Pipeline deadlock from ring undersizing | High | Low (validated) | `ring_size >= downstream_capacity + 2` assertion |

---

*End of Technical Audit*
