//! Upscale backend trait — the GPU-only inference contract.
//!
//! Every backend implementation must satisfy:
//!
//! 1. **GPU-resident I/O**: `process()` accepts and returns [`GpuTexture`].
//!    No host staging buffers, no CPU tensors, no implicit copies.
//!
//! 2. **Pre-allocated buffers**: output GPU memory should be allocated once
//!    during `initialize()` and reused across frames.  Per-frame allocation
//!    is a contract violation.
//!
//! 3. **Thread safety**: the trait object is `Send + Sync`.  Concurrent
//!    `process()` calls are *not* required — the pipeline serializes
//!    inference.  But the backend must tolerate being moved across threads
//!    by the async executor.
//!
//! 4. **Deterministic cleanup**: `shutdown()` releases GPU resources
//!    in a defined order.  Backends must also implement [`Drop`] as a
//!    safety net, but callers should prefer explicit `shutdown()`.

use async_trait::async_trait;

use crate::core::types::GpuTexture;
use crate::error::Result;

/// Metadata extracted from an ONNX model's input/output tensor descriptors.
///
/// Used by the pipeline to validate frame dimensions and pre-allocate
/// correctly sized GPU buffers before the first frame arrives.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// Human-readable model identifier (from ONNX metadata or filename).
    pub name: String,

    /// Spatial upscale factor (e.g. 2, 4).
    /// Determined by comparing model input and output spatial dimensions.
    pub scale: u32,

    /// Expected input tensor name in the ONNX graph (e.g. `"input"`).
    pub input_name: String,

    /// Expected output tensor name in the ONNX graph (e.g. `"output"`).
    pub output_name: String,

    /// Input channels (almost always 3 for RGB).
    pub input_channels: u32,

    /// Minimum supported input spatial dimensions.
    /// Models with dynamic axes report `(1, 1)`.
    pub min_input_hw: (u32, u32),

    /// Maximum supported input spatial dimensions.
    /// Models with dynamic axes report `(u32::MAX, u32::MAX)`.
    pub max_input_hw: (u32, u32),
}

/// GPU-only super-resolution inference backend.
///
/// See module-level documentation for the full contract.
#[async_trait]
pub trait UpscaleBackend: Send + Sync {
    /// Load the model, allocate GPU resources, build TensorRT engine cache.
    ///
    /// Must be called exactly once before `process()`.  Calling `initialize()`
    /// on an already-initialized backend is an error.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Inference`] if the model cannot be loaded or
    /// the execution provider rejects the graph.
    async fn initialize(&self) -> Result<()>;

    /// Run super-resolution inference on a single GPU-resident frame.
    ///
    /// # Input contract
    ///
    /// - `input.format` must be [`PixelFormat::RgbPlanarF32`].
    /// - `input.data` must reside on the same CUDA device as the backend.
    /// - Spatial dimensions must be within `ModelMetadata::{min,max}_input_hw`.
    ///
    /// # Output contract
    ///
    /// - Returned `GpuTexture` has format [`PixelFormat::RgbPlanarF32`].
    /// - Spatial dimensions are `(input.width × scale, input.height × scale)`.
    /// - Data resides on the same device, same CUDA context.
    /// - The output buffer is owned by the backend and reused across calls;
    ///   the caller must consume or clone it before the next `process()` call.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::NotInitialized`] if `initialize()` was not called.
    /// Returns [`EngineError::FormatMismatch`] on wrong input format.
    async fn process(&self, input: GpuTexture) -> Result<GpuTexture>;

    /// Release all GPU resources (ORT session, device buffers, streams).
    ///
    /// After `shutdown()`, any subsequent `process()` call returns
    /// [`EngineError::NotInitialized`].
    async fn shutdown(&self) -> Result<()>;

    /// Query model metadata without running inference.
    ///
    /// Available after `initialize()`.
    fn metadata(&self) -> Result<&ModelMetadata>;
}
