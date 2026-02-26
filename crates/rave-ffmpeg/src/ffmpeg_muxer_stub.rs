#![allow(missing_docs)]
//! Stub FFmpeg muxer for builds without FFmpeg runtime bindings.

use rave_core::codec_traits::BitstreamSink;
use rave_core::error::{EngineError, Result};

/// Stub container muxer used when FFmpeg runtime support is disabled.
pub struct FfmpegMuxer;

impl FfmpegMuxer {
    pub fn new(
        path: &std::path::Path,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
    ) -> Result<Self> {
        let _ = (path, width, height, fps_num, fps_den);
        Err(EngineError::Mux(
            "rave-ffmpeg built without `ffmpeg-runtime`; container mux is unavailable".into(),
        ))
    }
}

impl BitstreamSink for FfmpegMuxer {
    fn write_packet(
        &mut self,
        _data: &[u8],
        _pts: i64,
        _dts: i64,
        _is_keyframe: bool,
    ) -> Result<()> {
        Err(EngineError::Mux(
            "rave-ffmpeg built without `ffmpeg-runtime`; container mux is unavailable".into(),
        ))
    }

    fn flush(&mut self) -> Result<()> {
        Err(EngineError::Mux(
            "rave-ffmpeg built without `ffmpeg-runtime`; container mux is unavailable".into(),
        ))
    }
}
