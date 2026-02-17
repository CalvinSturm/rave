//! Container metadata probing via FFmpeg's `avformat`.
//!
//! Opens a container file (MP4, MKV, MOV, etc.), finds the best video
//! stream, and extracts codec/resolution/framerate metadata needed to
//! configure the decode pipeline.

use std::ptr;

use ffmpeg_sys_next::*;

use crate::codecs::sys::cudaVideoCodec;
use crate::error::{EngineError, Result};
use crate::io::ffmpeg_sys::{check_ffmpeg, to_cstring};

/// Metadata extracted from a container's video stream.
#[derive(Debug, Clone)]
pub struct ContainerMetadata {
    /// Video codec for NVDEC.
    pub codec: cudaVideoCodec,
    /// Coded width in pixels.
    pub width: u32,
    /// Coded height in pixels.
    pub height: u32,
    /// Framerate numerator.
    pub fps_num: u32,
    /// Framerate denominator.
    pub fps_den: u32,
    /// Stream time base (for PTS rescaling).
    pub time_base: AVRational,
    /// Duration in microseconds (0 if unknown).
    pub duration_us: i64,
}

/// RAII guard for `AVFormatContext` — ensures cleanup on all exit paths.
struct FormatGuard {
    ctx: *mut AVFormatContext,
}

impl Drop for FormatGuard {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: ctx was allocated by avformat_open_input.
            unsafe {
                avformat_close_input(&mut self.ctx);
            }
        }
    }
}

/// Probe a container file and return video stream metadata.
pub fn probe_container(path: &std::path::Path) -> Result<ContainerMetadata> {
    let path_str = path
        .to_str()
        .ok_or_else(|| EngineError::Probe("Non-UTF8 path".into()))?;
    let c_path = to_cstring(path_str).map_err(|e| EngineError::Probe(format!("{e}")))?;

    let mut fmt_ctx: *mut AVFormatContext = ptr::null_mut();

    // Open the container.
    // SAFETY: c_path is a valid null-terminated C string. fmt_ctx is an output.
    let ret = unsafe {
        avformat_open_input(
            &mut fmt_ctx,
            c_path.as_ptr(),
            ptr::null(),
            ptr::null_mut(),
        )
    };
    check_ffmpeg(ret, "avformat_open_input")
        .map_err(|e| EngineError::Probe(format!("{e}")))?;

    let guard = FormatGuard { ctx: fmt_ctx };

    // Find stream info.
    // SAFETY: fmt_ctx is valid (open succeeded).
    let ret = unsafe { avformat_find_stream_info(guard.ctx, ptr::null_mut()) };
    check_ffmpeg(ret, "avformat_find_stream_info")
        .map_err(|e| EngineError::Probe(format!("{e}")))?;

    // Find best video stream.
    let stream_index = unsafe {
        av_find_best_stream(
            guard.ctx,
            AVMediaType::AVMEDIA_TYPE_VIDEO,
            -1,
            -1,
            ptr::null_mut(),
            0,
        )
    };
    if stream_index < 0 {
        return Err(EngineError::Probe(
            "No video stream found in container".into(),
        ));
    }

    // Extract stream parameters.
    let stream = unsafe {
        let streams = (*guard.ctx).streams;
        &*(*streams.add(stream_index as usize))
    };
    let codecpar = unsafe { &*stream.codecpar };

    let codec = match codecpar.codec_id {
        AVCodecID::AV_CODEC_ID_H264 => cudaVideoCodec::H264,
        AVCodecID::AV_CODEC_ID_HEVC => cudaVideoCodec::HEVC,
        AVCodecID::AV_CODEC_ID_VP9 => cudaVideoCodec::VP9,
        AVCodecID::AV_CODEC_ID_AV1 => cudaVideoCodec::AV1,
        other => {
            return Err(EngineError::Probe(format!(
                "Unsupported video codec: {other:?}"
            )));
        }
    };

    // Determine framerate: prefer stream avg_frame_rate, fall back to r_frame_rate.
    let fps = if stream.avg_frame_rate.den > 0 && stream.avg_frame_rate.num > 0 {
        stream.avg_frame_rate
    } else {
        stream.r_frame_rate
    };
    let fps_num = if fps.num > 0 { fps.num as u32 } else { 30 };
    let fps_den = if fps.den > 0 { fps.den as u32 } else { 1 };

    // Duration: container duration is in AV_TIME_BASE (microseconds).
    let duration_us = unsafe {
        if (*guard.ctx).duration > 0 {
            (*guard.ctx).duration
        } else {
            0
        }
    };

    tracing::info!(
        path = %path.display(),
        ?codec,
        width = codecpar.width,
        height = codecpar.height,
        fps = format!("{fps_num}/{fps_den}"),
        duration_s = format!("{:.2}", duration_us as f64 / 1_000_000.0),
        "Probed container"
    );

    Ok(ContainerMetadata {
        codec,
        width: codecpar.width as u32,
        height: codecpar.height as u32,
        fps_num,
        fps_den,
        time_base: stream.time_base,
        duration_us,
    })
}
