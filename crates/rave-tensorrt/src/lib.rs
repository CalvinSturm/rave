#![doc = include_str!("../README.md")]

#[cfg(feature = "tensorrt-runtime")]
pub mod tensorrt;
#[cfg(not(feature = "tensorrt-runtime"))]
#[path = "tensorrt_stub.rs"]
pub mod tensorrt;

pub use tensorrt::{BatchConfig, TensorRtBackend, validate_batch_config};
