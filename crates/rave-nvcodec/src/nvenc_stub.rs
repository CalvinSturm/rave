//! CPU-only stub for builds without NVDEC/NVENC system dependencies.

use rave_core::codec_traits::{BitstreamSink, FrameEncoder};
use rave_core::error::{EngineError, Result};
use rave_core::types::FrameEnvelope;

/// Encoder configuration parameters.
#[derive(Clone, Debug)]
pub struct NvEncConfig {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub bitrate: u32,
    pub max_bitrate: u32,
    pub gop_length: u32,
    pub b_frames: u32,
    pub nv12_pitch: u32,
}

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
