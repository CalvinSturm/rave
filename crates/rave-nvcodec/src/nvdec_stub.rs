#![allow(missing_docs)]
//! CPU-only stub for builds without NVDEC/NVENC system dependencies.

use std::sync::Arc;

use rave_core::codec_traits::{BitstreamSource, DecodedFrame, FrameDecoder};
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec as CoreCodec;

/// Stub NVDEC decoder used when `rave_nvcodec_stub` cfg is active.
pub struct NvDecoder;

impl NvDecoder {
    pub fn new(
        ctx: Arc<GpuContext>,
        source: Box<dyn BitstreamSource>,
        codec: CoreCodec,
    ) -> Result<Self> {
        let _ = (ctx, source, codec);
        Err(EngineError::Decode(
            "rave-nvcodec built in stub mode: NVDEC is unavailable on this build host".into(),
        ))
    }
}

impl FrameDecoder for NvDecoder {
    fn decode_next(&mut self) -> Result<Option<DecodedFrame>> {
        Err(EngineError::Decode(
            "rave-nvcodec built in stub mode: NVDEC is unavailable at runtime".into(),
        ))
    }
}
