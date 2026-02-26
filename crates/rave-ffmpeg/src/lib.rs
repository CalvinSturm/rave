#![doc = include_str!("../README.md")]

/// FFmpeg-based demuxer: extracts compressed packets from MP4/MKV containers.
#[cfg(feature = "ffmpeg-runtime")]
pub mod ffmpeg_demuxer;
#[cfg(not(feature = "ffmpeg-runtime"))]
#[path = "ffmpeg_demuxer_stub.rs"]
pub mod ffmpeg_demuxer;
/// FFmpeg-based muxer: wraps encoded HEVC packets into MP4/MKV containers.
#[cfg(feature = "ffmpeg-runtime")]
pub mod ffmpeg_muxer;
#[cfg(not(feature = "ffmpeg-runtime"))]
#[path = "ffmpeg_muxer_stub.rs"]
pub mod ffmpeg_muxer;
/// FFI helpers: BSF declarations, FFmpeg error translation, and string utilities.
#[cfg(feature = "ffmpeg-runtime")]
pub mod ffmpeg_sys;
#[cfg(not(feature = "ffmpeg-runtime"))]
#[path = "ffmpeg_sys_stub.rs"]
pub mod ffmpeg_sys;
/// File-based [`rave_core::codec_traits::BitstreamSink`] for raw Annex B output.
pub mod file_sink;
/// File-based [`rave_core::codec_traits::BitstreamSource`] for raw Annex B input.
pub mod file_source;
/// Container probe: extracts codec, resolution, and framerate metadata via FFmpeg.
#[cfg(feature = "ffmpeg-runtime")]
pub mod probe;
#[cfg(not(feature = "ffmpeg-runtime"))]
#[path = "probe_stub.rs"]
pub mod probe;

pub use probe::{ContainerMetadata, probe_container};

#[cfg(test)]
mod tests {
    #[test]
    fn demux_and_mux_stay_at_packet_boundary() {
        let demux = include_str!("ffmpeg_demuxer.rs");
        let mux = include_str!("ffmpeg_muxer.rs");

        for forbidden in [
            "avcodec_send_packet",
            "avcodec_receive_frame",
            "AVFrame",
            "sws_scale",
            "rawvideo",
        ] {
            assert!(
                !demux.contains(forbidden),
                "demux path should not include raw frame API `{forbidden}`"
            );
            assert!(
                !mux.contains(forbidden),
                "mux path should not include raw frame API `{forbidden}`"
            );
        }
    }
}
