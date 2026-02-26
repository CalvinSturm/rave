//! File-based [`BitstreamSink`] — writes raw HEVC bitstream to a file.
//!
//! # Design
//!
//! Writes encoded HEVC NAL units directly to a file as Annex B bitstream.
//! The output can be played by ffplay or remuxed into MP4 via:
//!
//! ```bash
//! ffmpeg -i output.265 -c copy output.mp4
//! ```
//!
//! This is the MVP sink.  A proper muxer (MP4/MKV container) should
//! replace this in Phase 2+ for timestamp-accurate container output.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use rave_core::codec_traits::BitstreamSink;
use rave_core::error::{EngineError, Result};

/// Writes raw HEVC Annex B bitstream to a file.
pub struct FileBitstreamSink {
    writer: BufWriter<File>,
    bytes_written: u64,
    packets_written: u64,
    path: PathBuf,
}

impl FileBitstreamSink {
    pub fn new(path: PathBuf) -> Result<Self> {
        let file = File::create(&path).map_err(|e| {
            EngineError::Pipeline(format!(
                "Failed to create output file {}: {}",
                path.display(),
                e
            ))
        })?;

        tracing::info!(path = %path.display(), "Output bitstream sink opened");

        Ok(Self {
            writer: BufWriter::with_capacity(4 * 1024 * 1024, file), // 4 MiB buffer
            bytes_written: 0,
            packets_written: 0,
            path,
        })
    }
}

impl BitstreamSink for FileBitstreamSink {
    fn write_packet(
        &mut self,
        data: &[u8],
        _pts: i64,
        _dts: i64,
        _is_keyframe: bool,
    ) -> Result<()> {
        self.writer.write_all(data).map_err(|e| {
            EngineError::Mux(format!("Failed to write to {}: {}", self.path.display(), e))
        })?;

        self.bytes_written += data.len() as u64;
        self.packets_written += 1;

        if self.packets_written.is_multiple_of(100) {
            tracing::debug!(
                packets = self.packets_written,
                bytes_mb = self.bytes_written / (1024 * 1024),
                "Sink progress"
            );
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush().map_err(|e| {
            EngineError::Mux(format!("Failed to flush {}: {}", self.path.display(), e))
        })?;

        tracing::info!(
            path = %self.path.display(),
            packets = self.packets_written,
            bytes_mb = self.bytes_written / (1024 * 1024),
            "Sink flushed — encoding complete"
        );

        Ok(())
    }
}
