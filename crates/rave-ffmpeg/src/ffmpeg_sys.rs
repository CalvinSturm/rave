//! FFmpeg FFI helpers — error translation, safe string conversion, and
//! BSF (bitstream filter) FFI declarations missing from `ffmpeg-sys-next` v8.

use std::ffi::CString;
use std::fmt::{Display, Formatter};

// ── BSF FFI ──────────────────────────────────────────────────────────────────
//
// `ffmpeg-sys-next` v8 does not generate bindings for `libavcodec/bsf.h`,
// even though the API is still public in FFmpeg 8.x.  We declare the subset
// we need here so the demuxer can do MP4 → Annex B conversion.

/// Opaque bitstream filter descriptor (read-only, returned by FFmpeg).
#[repr(C)]
pub struct AVBitStreamFilter {
    _opaque: [u8; 0],
}

/// Bitstream filter instance.  Layout mirrors `libavcodec/bsf.h`.
#[repr(C)]
pub struct AVBSFContext {
    pub av_class: *const std::ffi::c_void,
    pub filter: *const AVBitStreamFilter,
    pub priv_data: *mut std::ffi::c_void,
    pub par_in: *mut ffmpeg_sys_next::AVCodecParameters,
    pub par_out: *mut ffmpeg_sys_next::AVCodecParameters,
    pub time_base_in: ffmpeg_sys_next::AVRational,
    pub time_base_out: ffmpeg_sys_next::AVRational,
}

unsafe extern "C" {
    pub fn av_bsf_get_by_name(name: *const std::ffi::c_char) -> *const AVBitStreamFilter;
    pub fn av_bsf_alloc(
        filter: *const AVBitStreamFilter,
        ctx: *mut *mut AVBSFContext,
    ) -> std::ffi::c_int;
    pub fn av_bsf_init(ctx: *mut AVBSFContext) -> std::ffi::c_int;
    pub fn av_bsf_send_packet(
        ctx: *mut AVBSFContext,
        pkt: *const ffmpeg_sys_next::AVPacket,
    ) -> std::ffi::c_int;
    pub fn av_bsf_receive_packet(
        ctx: *mut AVBSFContext,
        pkt: *mut ffmpeg_sys_next::AVPacket,
    ) -> std::ffi::c_int;
    pub fn av_bsf_free(ctx: *mut *mut AVBSFContext);
}

/// Structured FFmpeg error details for module-specific wrapping.
#[derive(Debug, Clone)]
pub struct FfmpegErrorDetail {
    /// Human-readable description of the operation that failed (e.g. `"avformat_open_input"`).
    pub context: String,
    /// Raw FFmpeg error code (negative AVERROR value).
    pub code: i32,
    /// Human-readable error message from `av_strerror`.
    pub message: String,
}

impl Display for FfmpegErrorDetail {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} (code {})", self.context, self.message, self.code)
    }
}

/// Translate an FFmpeg return code into a structured error.
///
/// On success (`ret >= 0`) this is a no-op. On failure, `av_strerror` is
/// called to produce a human-readable message.
pub fn check_ffmpeg(ret: i32, context: &str) -> std::result::Result<(), FfmpegErrorDetail> {
    if ret >= 0 {
        return Ok(());
    }

    let mut buf = [0 as std::ffi::c_char; 256];
    // SAFETY: buf is a valid mutable buffer of known length.
    unsafe {
        ffmpeg_sys_next::av_strerror(ret, buf.as_mut_ptr(), buf.len());
    }
    // Convert C string buffer to UTF-8 string.
    let msg = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_str()
        .unwrap_or("unknown error")
        .to_string();

    Err(FfmpegErrorDetail {
        context: context.to_string(),
        code: ret,
        message: msg,
    })
}

/// Convert a Rust `&str` to a `CString`, mapping NUL bytes to an error.
pub fn to_cstring(s: &str) -> std::result::Result<CString, String> {
    CString::new(s).map_err(|e| format!("Invalid path string: {e}"))
}
