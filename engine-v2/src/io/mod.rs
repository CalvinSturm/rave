//! I/O adapters for the pipeline.
//!
//! - [`FileBitstreamSource`] — reads raw H.264/HEVC bitstream from a file.
//! - [`FileBitstreamSink`] — writes encoded HEVC bitstream packets to a file.
//! - [`FfmpegDemuxer`] — reads video from containers (MP4/MKV/MOV) via FFmpeg.
//! - [`FfmpegMuxer`] — writes video to containers via FFmpeg.
//! - [`probe_container`] — probes container metadata (codec, resolution, fps).

pub mod ffmpeg_demuxer;
pub mod ffmpeg_muxer;
pub mod ffmpeg_sys;
pub mod file_sink;
pub mod file_source;
pub mod probe;

pub use probe::{probe_container, ContainerMetadata};
