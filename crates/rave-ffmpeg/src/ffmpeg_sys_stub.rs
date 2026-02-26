#![allow(missing_docs)]
//! Stub FFI helpers for builds without FFmpeg runtime bindings.

use std::ffi::CString;

/// Placeholder for FFmpeg's `AVBitStreamFilter`.
#[derive(Debug, Clone, Copy)]
pub struct AVBitStreamFilter;

/// Placeholder for FFmpeg's `AVBSFContext`.
#[derive(Debug, Clone, Copy)]
pub struct AVBSFContext;

/// Placeholder for FFmpeg's `AVRational`.
#[derive(Debug, Clone, Copy)]
pub struct AVRational {
    pub num: i32,
    pub den: i32,
}

/// Structured FFmpeg error context.
#[derive(Debug, Clone)]
pub struct FfmpegErrorDetail {
    pub context: String,
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for FfmpegErrorDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code {}): {}", self.context, self.code, self.message)
    }
}

impl std::error::Error for FfmpegErrorDetail {}

/// Stubbed FFmpeg status checker.
pub fn check_ffmpeg(ret: i32, context: &str) -> std::result::Result<(), FfmpegErrorDetail> {
    if ret >= 0 {
        Ok(())
    } else {
        Err(FfmpegErrorDetail {
            context: context.to_string(),
            code: ret,
            message: "FFmpeg runtime support is disabled in this build".into(),
        })
    }
}

/// Convert a Rust string to a C string.
pub fn to_cstring(s: &str) -> std::result::Result<CString, String> {
    CString::new(s).map_err(|e| e.to_string())
}
