#![allow(missing_docs)]
//! Stub FFmpeg demuxer for builds without FFmpeg runtime bindings.

use rave_core::codec_traits::{BitstreamPacket, BitstreamSource};
use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec;

/// Stub container demuxer used when FFmpeg runtime support is disabled.
pub struct FfmpegDemuxer;

impl FfmpegDemuxer {
    pub fn new(path: &std::path::Path, codec: cudaVideoCodec) -> Result<Self> {
        let _ = (path, codec);
        Err(EngineError::Demux(
            "rave-ffmpeg built without `ffmpeg-runtime`; container demux is unavailable".into(),
        ))
    }
}

impl BitstreamSource for FfmpegDemuxer {
    fn read_packet(&mut self) -> Result<Option<BitstreamPacket>> {
        Err(EngineError::Demux(
            "rave-ffmpeg built without `ffmpeg-runtime`; container demux is unavailable".into(),
        ))
    }
}
