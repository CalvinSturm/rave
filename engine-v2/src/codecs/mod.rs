//! GPU hardware codec integration — NVDEC + NVENC.
//!
//! - [`nvdec`] — Hardware video decoder (NV12 output to GPU memory)
//! - [`nvenc`] — Hardware video encoder (NV12 input from GPU memory)
//! - [`sys`] — Raw FFI bindings to nvcuvid + nvEncodeAPI

pub mod nvdec;
pub mod nvenc;
pub mod sys;
