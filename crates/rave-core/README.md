[![Crates.io](https://img.shields.io/crates/v/rave-core.svg)](https://crates.io/crates/rave-core)
[![docs.rs](https://docs.rs/rave-core/badge.svg)](https://docs.rs/rave-core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

# rave-core

Core types, traits, context, and error model shared by all RAVE crates.

`rave-core` defines the engine contracts used across decode, preprocess,
inference, and encode stages. It is the foundation crate for pipeline assembly.

## Scope

- GPU context and memory pooling (`GpuContext`, `PoolStats`, `HealthSnapshot`)
- Frame and pixel contracts (`GpuTexture`, `FrameEnvelope`, `PixelFormat`)
- Backend trait for inference (`UpscaleBackend`, `ModelMetadata`)
- Codec I/O traits (`BitstreamSource`, `BitstreamSink`, `FrameDecoder`, `FrameEncoder`)
- Unified error hierarchy (`EngineError`, `Result`)
- Low-level CUDA/NVDEC/NVENC FFI aliases used by higher-level crates

## Key Modules

- `context`: CUDA device/stream ownership, pooled allocations, VRAM accounting
- `types`: GPU-resident frame and pixel format contracts
- `backend`: inference backend trait + model metadata contract
- `codec_traits`: decode/encode source/sink abstractions
- `error`: typed errors with stable error codes
- `ffi_types`: C ABI type aliases and enums used by codec crates

## Feature Flags

- `debug-alloc`: enables host-allocation instrumentation helpers in `debug_alloc`

## Runtime Notes

- On Linux/WSL, runtime must resolve NVIDIA driver libraries (`libcuda.so`, etc.).
- This crate is shared by both runtime code and tests, so linker/runtime setup
  for CUDA affects downstream workspace crates.

## Minimal Example

```rust,no_run
use rave_core::context::GpuContext;
use rave_core::error::Result;

fn init() -> Result<()> {
    let ctx = GpuContext::new(0)?;
    let (_current, _peak) = ctx.vram_usage();
    Ok(())
}
```

## Relationship to Other Crates

- `rave-cuda`: implements CUDA kernels and stream helpers using `rave-core` types
- `rave-tensorrt`: implements `UpscaleBackend`
- `rave-nvcodec`: implements `FrameDecoder` and `FrameEncoder`
- `rave-pipeline`: orchestrates stage concurrency with these shared contracts
