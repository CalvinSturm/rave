#![doc = include_str!("../README.md")]
//!
//! # Stub mode
//!
//! When the `rave_nvcodec_stub` cfg is active (set by `build.rs` when the
//! NVIDIA Video Codec SDK headers or libraries are not found), the `nvdec`
//! and `nvenc` modules are replaced with stubs that return
//! [`EngineError::Decode`](rave_core::error::EngineError::Decode) /
//! [`EngineError::Encode`](rave_core::error::EngineError::Encode) immediately.
//! This allows the workspace to build and run tests on CI runners without a
//! GPU or NVIDIA drivers installed.

pub mod config;

#[cfg(rave_nvcodec_stub)]
#[path = "nvdec_stub.rs"]
pub mod nvdec;
#[cfg(not(rave_nvcodec_stub))]
pub mod nvdec;

#[cfg(rave_nvcodec_stub)]
#[path = "nvenc_stub.rs"]
pub mod nvenc;
#[cfg(not(rave_nvcodec_stub))]
pub mod nvenc;

pub mod sys;
