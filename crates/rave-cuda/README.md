[![Crates.io](https://img.shields.io/crates/v/rave-cuda.svg)](https://crates.io/crates/rave-cuda)
[![docs.rs](https://docs.rs/rave-cuda/badge.svg)](https://docs.rs/rave-cuda)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

# rave-cuda

CUDA kernel layer for RAVE preprocessing and postprocessing.

`rave-cuda` compiles and launches kernels used for frame conversion around
inference: NV12 -> RGB planar and RGB planar -> NV12, including F16/F32
precision paths.

## Scope

- NVRTC kernel compilation at startup (`PreprocessKernels::compile`)
- Launch helpers for format conversion and precision conversion
- End-to-end preprocess wrapper (`PreprocessPipeline`)
- Stream/event helpers for cross-stream ordering (`stream` module)

## Public API Highlights

- `PreprocessKernels`: compiled kernel handles
- `PreprocessPipeline`: precision-aware preprocess/postprocess chain
- `ModelInput`: model-ready tensor descriptor + shape metadata
- `ModelPrecision`: `F32` or `F16`
- `StageMetrics`: lightweight per-stage timing counters

## Typical Usage

```rust,no_run
use std::sync::Arc;

use rave_core::context::GpuContext;
use rave_core::error::Result;
use rave_cuda::kernels::{ModelPrecision, PreprocessKernels, PreprocessPipeline};

fn build_pipeline(ctx: Arc<GpuContext>) -> Result<PreprocessPipeline> {
    let kernels = PreprocessKernels::compile(ctx.device())?;
    Ok(PreprocessPipeline::new(kernels, ModelPrecision::F16))
}
```

## Runtime Requirements

- CUDA driver/runtime available to the linker/runtime environment
- NVRTC support (for startup kernel compilation)
- Compatible `cudarc` + CUDA toolkit version

## Notes

- Data stays in GPU memory; no host staging is done by this crate.
- Kernel launches are stream-ordered. Synchronization policy is controlled by
  caller orchestration (typically `rave-pipeline`).
