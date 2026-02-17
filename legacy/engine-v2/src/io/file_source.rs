//! File-based [`BitstreamSource`] — reads raw H.264/HEVC Annex B bitstream.
//!
//! # Design
//!
//! Reads the raw file into a single buffer and feeds it as one large packet.
//! NVDEC's parser handles NAL boundary detection internally.
//!
//! For container formats (MP4, MKV), a proper demuxer is needed (Phase 2+).
//! This implementation expects raw Annex B bitstream (`.264`, `.265`, `.hevc`).

use std::fs;
use std::path::PathBuf;

use crate::codecs::nvdec::{BitstreamPacket, BitstreamSource};
use crate::error::{EngineError, Result};

/// Reads a raw bitstream file and feeds it to NVDEC.
///
/// The entire file is loaded into memory on first `read_packet()` call.
/// Subsequent calls return `None` (EOS).
pub struct FileBitstreamSource {
    _path: PathBuf,
    data: Option<Vec<u8>>,
    sent: bool,
}

impl FileBitstreamSource {
    pub fn new(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            return Err(EngineError::Pipeline(format!(
                "Input file not found: {}",
                path.display()
            )));
        }

        let data = fs::read(&path).map_err(|e| {
            EngineError::Pipeline(format!("Failed to read {}: {}", path.display(), e))
        })?;

        tracing::info!(
            path = %path.display(),
            size_mb = data.len() / (1024 * 1024),
            "Loaded bitstream file"
        );

        Ok(Self {
            _path: path,
            data: Some(data),
            sent: false,
        })
    }
}

impl BitstreamSource for FileBitstreamSource {
    fn read_packet(&mut self) -> Result<Option<BitstreamPacket>> {
        if self.sent {
            return Ok(None);
        }

        let data = self
            .data
            .take()
            .ok_or_else(|| EngineError::Pipeline("BitstreamSource data already consumed".into()))?;

        self.sent = true;

        Ok(Some(BitstreamPacket {
            data,
            pts: 0,
            is_keyframe: true,
        }))
    }
}
