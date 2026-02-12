//! VideoForge v2.0 — GPU-native super-resolution engine.
//!
//! # Architecture
//!
//! All frame data is strictly GPU-resident. The pipeline moves frames as:
//!
//! ```text
//! NVDEC → Device Preprocess → TensorRT (ORT) → NVENC
//! ```
//!
//! No host copies. No implicit allocations. No CPU fallbacks.
//!
//! # Module layout
//!
//! - [`core`] — GPU contract types, backend trait, CUDA context, preprocessing kernels
//! - [`backends`] — Concrete inference backend implementations (TensorRT via ORT)
//! - [`engine`] — Pipeline orchestration with bounded backpressure
//! - [`error`] — Typed error hierarchy

pub mod backends;
pub mod codecs;
pub mod core;
pub mod debug_alloc;
pub mod engine;
pub mod error;

#[cfg(feature = "debug-alloc")]
#[global_allocator]
static ALLOC: crate::debug_alloc::TrackingAllocator = crate::debug_alloc::TrackingAllocator;
