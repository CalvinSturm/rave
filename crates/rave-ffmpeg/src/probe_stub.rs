#![allow(missing_docs)]
//! Stub container probing for builds without FFmpeg runtime bindings.

use std::path::Path;

use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec;

use crate::ffmpeg_sys::AVRational;

/// Metadata extracted from a container's video stream.
#[derive(Debug, Clone)]
pub struct ContainerMetadata {
    pub codec: cudaVideoCodec,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub time_base: AVRational,
    pub duration_us: i64,
}

/// Stub probe function used when FFmpeg runtime support is disabled.
pub fn probe_container(path: &Path) -> Result<ContainerMetadata> {
    let _ = path;
    Err(EngineError::Probe(
        "rave-ffmpeg built without `ffmpeg-runtime`; container probing is unavailable".into(),
    ))
}
