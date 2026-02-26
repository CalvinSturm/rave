#![allow(missing_docs)]
//! CPU-only stub for builds without NVDEC/NVENC system dependencies.

use rave_core::codec_traits::{BitstreamSink, FrameEncoder};
use rave_core::error::{EngineError, Result};
use rave_core::types::FrameEnvelope;

pub use crate::config::NvEncConfig;

/// Stub NVENC encoder used when `rave_nvcodec_stub` cfg is active.
pub struct NvEncoder;

impl NvEncoder {
    pub fn new(
        cuda_context: *mut std::ffi::c_void,
        sink: Box<dyn BitstreamSink>,
        config: NvEncConfig,
    ) -> Result<Self> {
        let _ = (cuda_context, sink, config);
        Err(EngineError::Encode(
            "rave-nvcodec built in stub mode: NVENC is unavailable on this build host".into(),
        ))
    }
}

impl FrameEncoder for NvEncoder {
    fn encode(&mut self, frame: FrameEnvelope) -> Result<()> {
        let _ = frame;
        Err(EngineError::Encode(
            "rave-nvcodec built in stub mode: NVENC is unavailable at runtime".into(),
        ))
    }

    fn flush(&mut self) -> Result<()> {
        Err(EngineError::Encode(
            "rave-nvcodec built in stub mode: NVENC is unavailable at runtime".into(),
        ))
    }
}
