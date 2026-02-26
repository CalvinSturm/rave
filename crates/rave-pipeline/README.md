[![Crates.io](https://img.shields.io/crates/v/rave-pipeline.svg)](https://crates.io/crates/rave-pipeline)
[![docs.rs](https://docs.rs/rave-pipeline/badge.svg)](https://docs.rs/rave-pipeline)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

# rave-pipeline

Bounded async pipeline orchestration for RAVE.

`rave-pipeline` coordinates decode, preprocess, inference, and encode stages
with backpressure-aware channels and cancellation support.

## Scope

- Main orchestrator: `UpscalePipeline`
- Pipeline configuration: `PipelineConfig`
- Atomic stage counters and latency tracking: `PipelineMetrics`
- End-to-end inference helper: `InferencePipeline` (with `nvidia-inference`)
- Synthetic stress/audit helpers in `pipeline` module

## Stage Model

Concurrent tasks:
- decode (blocking)
- preprocess (async)
- inference + postprocess (async)
- encode (blocking)

Bounded channels enforce finite in-flight frame counts and upstream backpressure.

## Public API Highlights

- `UpscalePipeline::new(ctx, kernels, config)`
- `UpscalePipeline::run(decoder, backend, encoder)`
- `UpscalePipeline::run_graph(input, output, graph, profile, contract)` (with `nvidia-run-graph`)
- `PipelineConfig` capacities and model precision controls
- `PipelineMetrics` frame counters and latency aggregates
- `StageGraph`, `StageConfig`, `ProfilePreset`, `RunContract`, `PipelineReport`

## Feature Flags

- `cuda-pipeline` (default): enables the CUDA-backed pipeline runtime (`pipeline` module)
- `nvidia-inference`: enables `InferencePipeline` (TensorRT/CUDA-backed inference helper)
- `nvidia-run-graph`: enables concrete graph execution helpers that wire FFmpeg/NVDEC/NVENC
- `compat-runtime-nvidia`: temporary compatibility shim for `rave_pipeline::runtime` re-exports
- `audit-no-host-copies`: strict host-copy audit integration with `rave-core`/`rave-tensorrt`
- `debug-host-copies`: enables determinism/debug readback paths

## Typical Usage

```rust,no_run
use std::sync::Arc;

use rave_core::context::GpuContext;
use rave_core::error::Result;
use rave_cuda::kernels::{ModelPrecision, PreprocessKernels};
use rave_pipeline::pipeline::{PipelineConfig, UpscalePipeline};

fn build_pipeline(ctx: Arc<GpuContext>) -> Result<UpscalePipeline> {
    let kernels = Arc::new(PreprocessKernels::compile(ctx.device())?);
    let cfg = PipelineConfig {
        model_precision: ModelPrecision::F16,
        encoder_nv12_pitch: 256,
        ..PipelineConfig::default()
    };
    Ok(UpscalePipeline::new(ctx, kernels, cfg))
}
```

## Notes

- Cancellation is propagated via `CancellationToken`.
- Queue depth and metrics are designed for production telemetry.
- Inference backend is pluggable via `rave_core::backend::UpscaleBackend`.
- Optional strict no-host-copies checks are controlled by
  `PipelineConfig::strict_no_host_copies` with crate feature
  `audit-no-host-copies`.
- Determinism checkpoint hashing is feature-gated behind `debug-host-copies`.
