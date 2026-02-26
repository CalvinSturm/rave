[![Crates.io](https://img.shields.io/crates/v/rave-nvcodec.svg)](https://crates.io/crates/rave-nvcodec)
[![docs.rs](https://docs.rs/rave-nvcodec/badge.svg)](https://docs.rs/rave-nvcodec)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

# rave-nvcodec

NVDEC/NVENC hardware codec wrappers for RAVE.

`rave-nvcodec` implements GPU decode and encode stages over NVIDIA Video Codec
SDK APIs with CUDA-device-pointer based frame flow.

## Scope

- NVDEC decode wrapper (`NvDecoder`) implementing `FrameDecoder`
- NVENC encode wrapper (`NvEncoder`) implementing `FrameEncoder`
- Encoder configuration (`NvEncConfig`)
- Raw FFI bindings for CUDA Video Codec SDK (`sys` module)

## Public API Highlights

- `NvDecoder::new(ctx, source, codec)`
- `NvEncoder::new(cuda_context, sink, config)`
- `NvEncConfig` for resolution/fps/bitrate/GOP/B-frame/pitch settings

## Typical Usage

```rust,no_run
use std::sync::Arc;

use rave_core::codec_traits::{BitstreamSink, BitstreamSource};
use rave_core::context::GpuContext;
use rave_core::error::Result;
use rave_core::ffi_types::cudaVideoCodec;
use rave_nvcodec::nvdec::NvDecoder;

fn make_decoder(ctx: Arc<GpuContext>, source: Box<dyn BitstreamSource>) -> Result<NvDecoder> {
    NvDecoder::new(ctx, source, cudaVideoCodec::HEVC)
}
```

## Runtime Requirements

- NVIDIA driver libraries (`libcuda`, `libnvcuvid`, `libnvidia-encode`)
- Compatible CUDA toolkit + Video Codec SDK runtime
- Linux/WSL link search paths must include locations for NVIDIA driver libs

## Notes

- Decode output and encode input are GPU-resident NV12 frames.
- Encode path caches resource registration by device pointer to avoid
  per-frame registration overhead.
- This crate performs FFI-heavy work; all unsafe boundaries are isolated in
  codec modules and `sys`.
