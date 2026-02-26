#![doc = include_str!("../README.md")]

pub mod kernels;
pub mod stream;
mod sys;
pub use kernels::{
    ModelInput, ModelPrecision, PreprocessKernels, PreprocessPipeline, StageMetrics,
};
