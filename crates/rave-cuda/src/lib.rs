#![doc = include_str!("../README.md")]

pub mod blur;
pub mod kernels;
pub mod stream;
pub mod sys;

pub use blur::{BlurRegion, FaceBlurConfig, FaceBlurEngine};
pub use kernels::{
    ModelInput, ModelPrecision, PreprocessKernels, PreprocessPipeline, StageMetrics,
};
