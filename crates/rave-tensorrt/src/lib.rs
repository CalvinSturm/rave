#![doc = include_str!("../README.md")]

pub mod tensorrt;

pub use tensorrt::{BatchConfig, TensorRtBackend, validate_batch_config};
