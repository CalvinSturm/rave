//! Shared codec traits used across crate boundaries.
//!
//! These traits break the circular dependency between `rave-nvcodec`,
//! `rave-ffmpeg`, and `rave-pipeline` by providing a neutral home.

use crate::error::Result;
use crate::ffi_types::CUevent;
use crate::types::FrameEnvelope;

// ─── Bitstream source (demuxer → decoder) ────────────────────────────────

/// Demuxed compressed bitstream packets (host-side, NOT raw pixels).
///
/// Implementations: file reader, network receiver, FFmpeg demuxer, etc.
pub trait BitstreamSource: Send + 'static {
    /// Read the next compressed bitstream packet, or `None` at end-of-stream.
    fn read_packet(&mut self) -> Result<Option<BitstreamPacket>>;
}

/// A single demuxed NAL unit or access unit.
pub struct BitstreamPacket {
    /// Compressed bitstream data (Annex B or length-prefixed).
    /// Host memory is acceptable — this is codec-compressed (~10 KB/frame).
    pub data: Vec<u8>,
    /// Presentation timestamp in microseconds.
    ///
    /// Container-specific time bases are converted at the demux boundary
    /// (for example in `rave-ffmpeg`) so downstream decode/encode stages
    /// operate on one stable unit.
    pub pts: i64,
    /// Whether this packet encodes an IDR/keyframe.
    pub is_keyframe: bool,
}

// ─── Bitstream sink (encoder → muxer) ────────────────────────────────────

/// Receives encoded bitstream output.
///
/// Implementations: file writer, muxer, network sender, etc.
pub trait BitstreamSink: Send + 'static {
    /// Write one encoded packet.
    ///
    /// `pts` and `dts` are in microseconds. Container muxers are responsible
    /// for rescaling to stream time_base at the output boundary.
    fn write_packet(&mut self, data: &[u8], pts: i64, dts: i64, is_keyframe: bool) -> Result<()>;
    /// Flush any internal buffers and finalise the output stream.
    fn flush(&mut self) -> Result<()>;
}

// ─── Pipeline stage traits ───────────────────────────────────────────────

/// A decoded NV12 frame with an optional CUDA event for cross-stream sync.
///
/// When produced by a real hardware decoder (NVDEC), `decode_event` carries
/// the event recorded on `decode_stream` after the D2D copy.  The preprocess
/// stage calls `cuStreamWaitEvent(preprocess_stream, event)` before reading
/// the texture — this ensures the decode data is ready without CPU-blocking.
///
/// Mock decoders set both fields to `None`.
pub struct DecodedFrame {
    pub envelope: FrameEnvelope,
    /// CUDA event recorded after decode completes.  `None` for mock decoders.
    pub decode_event: Option<CUevent>,
    /// Channel to return used events to the decoder's `EventPool` for reuse.
    pub event_return: Option<std::sync::mpsc::Sender<CUevent>>,
}

// SAFETY: CUevent is a CUDA handle (*mut c_void) that is safe to send across
// threads — CUDA events have no thread affinity.
unsafe impl Send for DecodedFrame {}

/// Video frame decoder producing GPU-resident NV12 frames.
pub trait FrameDecoder: Send + 'static {
    /// Decode the next frame from the bitstream source.
    ///
    /// Returns `None` at end-of-stream.
    fn decode_next(&mut self) -> Result<Option<DecodedFrame>>;
}

/// Video frame encoder consuming GPU-resident NV12 frames.
pub trait FrameEncoder: Send + 'static {
    /// Submit one GPU-resident NV12 frame for encoding.
    fn encode(&mut self, frame: FrameEnvelope) -> Result<()>;
    /// Flush any pending frames and finalise the bitstream.
    fn flush(&mut self) -> Result<()>;
}

// ─── Model precision ─────────────────────────────────────────────────────

/// Which floating-point precision the inference model expects.
///
/// Mirrors [`rave_cuda::kernels::ModelPrecision`] — the two types are
/// identical and convert via pattern matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelPrecision {
    /// 32-bit single-precision float.
    F32,
    /// 16-bit half-precision float.
    F16,
}
