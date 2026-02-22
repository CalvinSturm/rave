#![doc = include_str!("../README.md")]

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
