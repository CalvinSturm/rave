[![Crates.io](https://img.shields.io/crates/v/rave-tensorrt.svg)](https://crates.io/crates/rave-tensorrt)
[![docs.rs](https://docs.rs/rave-tensorrt/badge.svg)](https://docs.rs/rave-tensorrt)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

# rave-tensorrt

TensorRT-backed inference implementation for RAVE.

`rave-tensorrt` provides an `UpscaleBackend` implementation powered by
ONNX Runtime with TensorRT/CUDA execution providers and device-memory I/O
binding.

## Scope

- Backend implementation: `TensorRtBackend`
- Precision policy controls: `PrecisionPolicy`
- Batch settings surface: `BatchConfig`
- Inference/output-ring metrics and telemetry
- Provider bridge handling for Linux/WSL runtime compatibility

## Public API Highlights

- `TensorRtBackend::new(model_path, ctx, device_id, ring_size, downstream_capacity)`
- `TensorRtBackend::with_precision(...)`
- `BatchConfig { max_batch, latency_deadline_us }`
- `TensorRtBackend` implements `rave_core::backend::UpscaleBackend`

## Typical Usage

```rust,no_run
use std::path::PathBuf;
use std::sync::Arc;

use rave_core::backend::UpscaleBackend;
use rave_core::context::GpuContext;
use rave_core::error::Result;
use rave_tensorrt::tensorrt::TensorRtBackend;

async fn init_backend(ctx: Arc<GpuContext>) -> Result<TensorRtBackend> {
    let backend = TensorRtBackend::new(PathBuf::from("model.onnx"), ctx, 0, 6, 4);
    backend.initialize().await?;
    Ok(backend)
}
```

## Runtime Requirements

- ONNX Runtime shared libs and provider libs discoverable at runtime
- CUDA/TensorRT/cuDNN stack compatible with ORT build
- NVIDIA driver libs discoverable (`libcuda`, etc.)

## Notes

- Data path is GPU-resident via ORT I/O binding.
- `BatchConfig` is part of the API surface; current processing path is single-frame.
- Provider selection can be controlled externally via `RAVE_ORT_TENSORRT`.
