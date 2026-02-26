//! Shared encoder configuration types used by both the real and stub NVENC backends.

/// Encoder configuration parameters.
///
/// Passed to [`NvEncoder::new`](crate::nvenc::NvEncoder::new) to configure the
/// NVENC session before the first frame is encoded.
#[derive(Clone, Debug)]
pub struct NvEncConfig {
    /// Target frame width in pixels.
    pub width: u32,
    /// Target frame height in pixels.
    pub height: u32,
    /// Framerate numerator.
    pub fps_num: u32,
    /// Framerate denominator.
    pub fps_den: u32,
    /// Average bitrate in bits/sec (`0` = CQP mode).
    pub bitrate: u32,
    /// Peak bitrate in bits/sec (VBR mode; `0` = unconstrained).
    pub max_bitrate: u32,
    /// GOP length — number of frames between IDR pictures.
    pub gop_length: u32,
    /// B-frame interval (`0` = no B-frames).
    pub b_frames: u32,
    /// NV12 row pitch in bytes. Must match the pitch of incoming
    /// [`GpuTexture`](rave_core::types::GpuTexture) frames exactly.
    pub nv12_pitch: u32,
}
