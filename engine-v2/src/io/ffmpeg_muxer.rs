//! FFmpeg-based container muxer — [`BitstreamSink`] impl for MP4/MKV/MOV.
//!
//! Writes encoded HEVC bitstream packets into a container file.
//! The output format is auto-detected from the file extension.

use std::ptr;

use ffmpeg_sys_next::*;

use crate::codecs::nvenc::BitstreamSink;
use crate::error::{EngineError, Result};
use crate::io::ffmpeg_sys::{check_ffmpeg, to_cstring};

/// Muxes encoded bitstream into a container file.
pub struct FfmpegMuxer {
    fmt_ctx: *mut AVFormatContext,
    stream: *mut AVStream,
    pkt: *mut AVPacket,
    /// Stream time_base (set after avformat_write_header).
    time_base: AVRational,
    /// Microsecond time_base for rescaling from pipeline convention.
    us_tb: AVRational,
    packet_counter: u64,
    header_written: bool,
}

// SAFETY: All FFmpeg operations happen on the encode blocking task.
unsafe impl Send for FfmpegMuxer {}

impl FfmpegMuxer {
    /// Create a muxer that writes to the given path.
    ///
    /// The container format is auto-detected from the file extension
    /// (`.mp4`, `.mkv`, `.mov`, etc.).
    pub fn new(
        path: &std::path::Path,
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
    ) -> Result<Self> {
        let path_str = path
            .to_str()
            .ok_or_else(|| EngineError::Mux("Non-UTF8 path".into()))?;
        let c_path = to_cstring(path_str).map_err(|e| EngineError::Mux(format!("{e}")))?;

        // ── Create output format context ──
        let mut fmt_ctx: *mut AVFormatContext = ptr::null_mut();
        let ret = unsafe {
            avformat_alloc_output_context2(
                &mut fmt_ctx,
                ptr::null(),
                ptr::null(),
                c_path.as_ptr(),
            )
        };
        if ret < 0 || fmt_ctx.is_null() {
            return Err(EngineError::Mux(format!(
                "Failed to create output context for {}",
                path.display()
            )));
        }

        // ── Add HEVC video stream ──
        let stream = unsafe { avformat_new_stream(fmt_ctx, ptr::null()) };
        if stream.is_null() {
            unsafe { avformat_free_context(fmt_ctx) };
            return Err(EngineError::Mux("Failed to create output stream".into()));
        }

        // Configure codec parameters.
        unsafe {
            let par = (*stream).codecpar;
            (*par).codec_type = AVMediaType::AVMEDIA_TYPE_VIDEO;
            (*par).codec_id = AVCodecID::AV_CODEC_ID_HEVC;
            (*par).width = width as i32;
            (*par).height = height as i32;
            // Set the stream time_base to match the input framerate.
            (*stream).time_base = AVRational {
                num: fps_den as i32,
                den: fps_num as i32,
            };
        }

        // ── Open avio ──
        let oformat = unsafe { (*fmt_ctx).oformat };
        let needs_file = unsafe { (*oformat).flags & AVFMT_NOFILE == 0 };
        if needs_file {
            let ret = unsafe { avio_open(&mut (*fmt_ctx).pb, c_path.as_ptr(), AVIO_FLAG_WRITE) };
            if ret < 0 {
                unsafe { avformat_free_context(fmt_ctx) };
                check_ffmpeg(ret, "avio_open")
                    .map_err(|e| EngineError::Mux(format!("{e}")))?;
            }
        }

        // ── Allocate packet ──
        let pkt = unsafe { av_packet_alloc() };
        if pkt.is_null() {
            unsafe {
                if needs_file {
                    avio_closep(&mut (*fmt_ctx).pb);
                }
                avformat_free_context(fmt_ctx);
            }
            return Err(EngineError::Mux("Failed to allocate AVPacket".into()));
        }

        tracing::info!(
            path = %path.display(),
            width,
            height,
            fps = format!("{fps_num}/{fps_den}"),
            "FFmpeg muxer opened"
        );

        Ok(Self {
            fmt_ctx,
            stream,
            pkt,
            time_base: AVRational { num: 0, den: 1 },
            us_tb: AVRational { num: 1, den: 1_000_000 },
            packet_counter: 0,
            header_written: false,
        })
    }

    /// Write the container header (lazily on first packet).
    fn write_header_if_needed(&mut self) -> Result<()> {
        if self.header_written {
            return Ok(());
        }

        let ret = unsafe { avformat_write_header(self.fmt_ctx, ptr::null_mut()) };
        check_ffmpeg(ret, "avformat_write_header")
            .map_err(|e| EngineError::Mux(format!("{e}")))?;

        // Capture the actual time_base after muxer initialization
        // (the muxer may adjust it).
        self.time_base = unsafe { (*self.stream).time_base };

        tracing::debug!(
            time_base_num = self.time_base.num,
            time_base_den = self.time_base.den,
            "Container header written"
        );

        self.header_written = true;
        Ok(())
    }
}

impl BitstreamSink for FfmpegMuxer {
    fn write_packet(&mut self, data: &[u8], pts: i64, is_keyframe: bool) -> Result<()> {
        self.write_header_if_needed()?;

        unsafe {
            // Copy data into the AVPacket.
            let ret = av_new_packet(self.pkt, data.len() as i32);
            if ret < 0 {
                check_ffmpeg(ret, "av_new_packet")
                    .map_err(|e| EngineError::Mux(format!("{e}")))?;
            }
            ptr::copy_nonoverlapping(data.as_ptr(), (*self.pkt).data, data.len());

            // Rescale PTS from microseconds to stream time_base.
            let stream_pts = av_rescale_q(pts, self.us_tb, self.time_base);
            (*self.pkt).pts = stream_pts;
            (*self.pkt).dts = stream_pts;
            (*self.pkt).stream_index = 0;
            // Duration = 1 tick in stream time_base (since time_base is fps_den/fps_num,
            // one tick = one frame).
            (*self.pkt).duration = 1;

            if is_keyframe {
                (*self.pkt).flags |= AV_PKT_FLAG_KEY;
            }

            let ret = av_interleaved_write_frame(self.fmt_ctx, self.pkt);
            // av_interleaved_write_frame takes ownership — unrefs internally.
            if ret < 0 {
                check_ffmpeg(ret, "av_interleaved_write_frame")
                    .map_err(|e| EngineError::Mux(format!("{e}")))?;
            }
        }

        self.packet_counter += 1;
        if self.packet_counter % 100 == 0 {
            tracing::debug!(packets = self.packet_counter, "Muxer progress");
        }

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if !self.header_written {
            // Nothing was written — no trailer needed.
            return Ok(());
        }

        let ret = unsafe { av_write_trailer(self.fmt_ctx) };
        check_ffmpeg(ret, "av_write_trailer")
            .map_err(|e| EngineError::Mux(format!("{e}")))?;

        tracing::info!(
            packets = self.packet_counter,
            "Container finalized — muxing complete"
        );

        Ok(())
    }
}

impl Drop for FfmpegMuxer {
    fn drop(&mut self) {
        unsafe {
            av_packet_free(&mut self.pkt);

            let oformat = (*self.fmt_ctx).oformat;
            if (*oformat).flags & AVFMT_NOFILE == 0 && !(*self.fmt_ctx).pb.is_null() {
                avio_closep(&mut (*self.fmt_ctx).pb);
            }

            avformat_free_context(self.fmt_ctx);
            self.fmt_ctx = ptr::null_mut();
        }
        tracing::debug!("FFmpeg muxer destroyed");
    }
}
