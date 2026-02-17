//! FFmpeg-based container demuxer — [`BitstreamSource`] impl for MP4/MKV/MOV.
//!
//! Reads compressed video packets from a container file and converts them
//! from MP4 length-prefixed format to Annex B NAL units via the appropriate
//! bitstream filter (`h264_mp4toannexb` or `hevc_mp4toannexb`).

use std::ptr;

use ffmpeg_sys_next::*;

/// POSIX EAGAIN — used with AVERROR() for "try again" semantics.
const EAGAIN: i32 = 11;

use crate::ffmpeg_sys::{
    // BSF FFI (missing from ffmpeg-sys-next v8)
    AVBSFContext,
    av_bsf_alloc,
    av_bsf_free,
    av_bsf_get_by_name,
    av_bsf_init,
    av_bsf_receive_packet,
    av_bsf_send_packet,
    check_ffmpeg,
    to_cstring,
};
use rave_core::codec_traits::{BitstreamPacket, BitstreamSource};
use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec;

/// Demuxes a container file and produces Annex B bitstream packets.
pub struct FfmpegDemuxer {
    fmt_ctx: *mut AVFormatContext,
    bsf_ctx: *mut AVBSFContext,
    video_stream_index: i32,
    /// Packet for reading from the container.
    pkt_read: *mut AVPacket,
    /// Packet for receiving filtered output from BSF.
    pkt_filtered: *mut AVPacket,
    /// Stream time_base for PTS rescaling to microseconds.
    time_base: AVRational,
    eos: bool,
}

// SAFETY: All FFmpeg operations happen on a single thread (the decode
// blocking task).  The raw pointers are not shared across threads.
unsafe impl Send for FfmpegDemuxer {}

impl FfmpegDemuxer {
    /// Open a container and prepare the Annex B bitstream filter.
    pub fn new(path: &std::path::Path, codec: cudaVideoCodec) -> Result<Self> {
        let path_str = path
            .to_str()
            .ok_or_else(|| EngineError::Demux("Non-UTF8 path".into()))?;
        let c_path = to_cstring(path_str)?;

        // ── Open container ──
        let mut fmt_ctx: *mut AVFormatContext = ptr::null_mut();
        let ret = unsafe {
            avformat_open_input(&mut fmt_ctx, c_path.as_ptr(), ptr::null(), ptr::null_mut())
        };
        check_ffmpeg(ret, "avformat_open_input")?;

        let ret = unsafe { avformat_find_stream_info(fmt_ctx, ptr::null_mut()) };
        if ret < 0 {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            check_ffmpeg(ret, "avformat_find_stream_info")?;
        }

        // ── Find video stream ──
        let stream_index = unsafe {
            av_find_best_stream(
                fmt_ctx,
                AVMediaType::AVMEDIA_TYPE_VIDEO,
                -1,
                -1,
                ptr::null_mut(),
                0,
            )
        };
        if stream_index < 0 {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            return Err(EngineError::Demux(
                "No video stream found in container".into(),
            ));
        }

        let stream = unsafe { &*(*(*fmt_ctx).streams.add(stream_index as usize)) };
        let time_base = stream.time_base;

        // ── Initialize bitstream filter (MP4 → Annex B) ──
        let bsf_name = match codec {
            cudaVideoCodec::H264 => c"h264_mp4toannexb",
            cudaVideoCodec::HEVC => c"hevc_mp4toannexb",
            _ => {
                // For VP9/AV1, no BSF needed — pass through raw.
                // We set bsf_ctx to null and skip filtering.
                let pkt_read = unsafe { av_packet_alloc() };
                let pkt_filtered = unsafe { av_packet_alloc() };
                if pkt_read.is_null() || pkt_filtered.is_null() {
                    unsafe { avformat_close_input(&mut fmt_ctx) };
                    return Err(EngineError::Demux("Failed to allocate AVPacket".into()));
                }

                tracing::info!(
                    path = %path.display(),
                    ?codec,
                    "FFmpeg demuxer opened (no BSF needed)"
                );

                return Ok(Self {
                    fmt_ctx,
                    bsf_ctx: ptr::null_mut(),
                    video_stream_index: stream_index,
                    pkt_read,
                    pkt_filtered,
                    time_base,
                    eos: false,
                });
            }
        };

        let bsf = unsafe { av_bsf_get_by_name(bsf_name.as_ptr()) };
        if bsf.is_null() {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            return Err(EngineError::BitstreamFilter(format!(
                "BSF {:?} not found — FFmpeg build may be incomplete",
                bsf_name
            )));
        }

        let mut bsf_ctx: *mut AVBSFContext = ptr::null_mut();
        let ret = unsafe { av_bsf_alloc(bsf, &mut bsf_ctx) };
        if ret < 0 {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            check_ffmpeg(ret, "av_bsf_alloc")?;
        }

        // Copy codec parameters from the stream to the BSF.
        let ret = unsafe { avcodec_parameters_copy((*bsf_ctx).par_in, stream.codecpar) };
        if ret < 0 {
            unsafe {
                av_bsf_free(&mut bsf_ctx);
                avformat_close_input(&mut fmt_ctx);
            }
            check_ffmpeg(ret, "avcodec_parameters_copy")?;
        }

        let ret = unsafe { av_bsf_init(bsf_ctx) };
        if ret < 0 {
            unsafe {
                av_bsf_free(&mut bsf_ctx);
                avformat_close_input(&mut fmt_ctx);
            }
            check_ffmpeg(ret, "av_bsf_init")?;
        }

        // ── Allocate packets ──
        let pkt_read = unsafe { av_packet_alloc() };
        let pkt_filtered = unsafe { av_packet_alloc() };
        if pkt_read.is_null() || pkt_filtered.is_null() {
            unsafe {
                av_bsf_free(&mut bsf_ctx);
                avformat_close_input(&mut fmt_ctx);
            }
            return Err(EngineError::Demux("Failed to allocate AVPacket".into()));
        }

        tracing::info!(
            path = %path.display(),
            ?codec,
            stream_index,
            "FFmpeg demuxer opened with Annex B BSF"
        );

        Ok(Self {
            fmt_ctx,
            bsf_ctx,
            video_stream_index: stream_index,
            pkt_read,
            pkt_filtered,
            time_base,
            eos: false,
        })
    }

    /// Rescale PTS from stream time_base to microseconds.
    fn rescale_pts(&self, pts: i64) -> i64 {
        if pts == AV_NOPTS_VALUE {
            return 0;
        }
        let us_tb = AVRational {
            num: 1,
            den: 1_000_000,
        };
        unsafe { av_rescale_q(pts, self.time_base, us_tb) }
    }

    fn copy_packet_data(pkt: &AVPacket) -> Result<Vec<u8>> {
        if pkt.size <= 0 {
            return Ok(Vec::new());
        }
        if pkt.data.is_null() {
            return Err(EngineError::Demux(
                "FFmpeg produced packet with null data pointer".into(),
            ));
        }
        // SAFETY: `pkt.data` is valid for `pkt.size` bytes when size > 0.
        Ok(unsafe { std::slice::from_raw_parts(pkt.data, pkt.size as usize) }.to_vec())
    }
}

impl BitstreamSource for FfmpegDemuxer {
    fn read_packet(&mut self) -> Result<Option<BitstreamPacket>> {
        if self.eos {
            return Ok(None);
        }

        loop {
            // If we have a BSF, try to receive a filtered packet first.
            if !self.bsf_ctx.is_null() {
                let ret = unsafe { av_bsf_receive_packet(self.bsf_ctx, self.pkt_filtered) };
                if ret == 0 {
                    let pkt = unsafe { &*self.pkt_filtered };
                    let data = Self::copy_packet_data(pkt)?;
                    let pts = self.rescale_pts(pkt.pts);
                    let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
                    unsafe { av_packet_unref(self.pkt_filtered) };
                    if data.is_empty() {
                        tracing::debug!("Skipping empty demuxed packet after BSF");
                        continue;
                    }
                    return Ok(Some(BitstreamPacket {
                        data,
                        pts,
                        is_keyframe,
                    }));
                } else if ret != AVERROR(EAGAIN) && ret != AVERROR_EOF {
                    check_ffmpeg(ret, "av_bsf_receive_packet")?;
                }
            }

            // Read the next frame from the container.
            let ret = unsafe { av_read_frame(self.fmt_ctx, self.pkt_read) };
            if ret < 0 {
                // EOF or error.
                self.eos = true;
                if ret == AVERROR_EOF {
                    // Flush the BSF.
                    if !self.bsf_ctx.is_null() {
                        unsafe { av_bsf_send_packet(self.bsf_ctx, ptr::null()) };
                        // Try to receive remaining packets.
                        let ret2 =
                            unsafe { av_bsf_receive_packet(self.bsf_ctx, self.pkt_filtered) };
                        if ret2 == 0 {
                            let pkt = unsafe { &*self.pkt_filtered };
                            let data = Self::copy_packet_data(pkt)?;
                            let pts = self.rescale_pts(pkt.pts);
                            let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
                            unsafe { av_packet_unref(self.pkt_filtered) };
                            if data.is_empty() {
                                return Ok(None);
                            }
                            return Ok(Some(BitstreamPacket {
                                data,
                                pts,
                                is_keyframe,
                            }));
                        }
                    }
                    return Ok(None);
                }
                check_ffmpeg(ret, "av_read_frame")?;
            }

            let pkt = unsafe { &*self.pkt_read };

            // Skip non-video streams.
            if pkt.stream_index != self.video_stream_index {
                unsafe { av_packet_unref(self.pkt_read) };
                continue;
            }

            if self.bsf_ctx.is_null() {
                // No BSF — return packet directly.
                let data = Self::copy_packet_data(pkt)?;
                let pts = self.rescale_pts(pkt.pts);
                let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
                unsafe { av_packet_unref(self.pkt_read) };
                if data.is_empty() {
                    tracing::debug!("Skipping empty demuxed packet (no BSF)");
                    continue;
                }
                return Ok(Some(BitstreamPacket {
                    data,
                    pts,
                    is_keyframe,
                }));
            }

            // Send packet to BSF for Annex B conversion.
            let ret = unsafe { av_bsf_send_packet(self.bsf_ctx, self.pkt_read) };
            unsafe { av_packet_unref(self.pkt_read) };
            if ret < 0 {
                check_ffmpeg(ret, "av_bsf_send_packet")?;
            }
            // Loop back to receive filtered output.
        }
    }
}

impl Drop for FfmpegDemuxer {
    fn drop(&mut self) {
        // Free in reverse allocation order.
        unsafe {
            av_packet_free(&mut self.pkt_filtered);
            av_packet_free(&mut self.pkt_read);
            if !self.bsf_ctx.is_null() {
                av_bsf_free(&mut self.bsf_ctx);
            }
            if !self.fmt_ctx.is_null() {
                avformat_close_input(&mut self.fmt_ctx);
            }
        }
        tracing::debug!("FFmpeg demuxer destroyed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_packet_data_is_safe() {
        let pkt: AVPacket = unsafe { std::mem::zeroed() };
        let data = FfmpegDemuxer::copy_packet_data(&pkt).expect("empty packet should be accepted");
        assert!(data.is_empty());
    }
}
