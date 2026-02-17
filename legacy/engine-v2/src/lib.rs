//! # RAVE — GPU Video Inference Pipeline
//!
//! `rave-engine` is a deterministic, production-grade GPU video super-resolution
//! engine written in Rust. It decodes video with NVDEC, runs ONNX model inference
//! through TensorRT, and encodes the upscaled output with NVENC — using GPU-resident
//! frame transport with no CPU frame copies in steady-state operation. The engine
//! targets hardware-accelerated video upscaling workloads where Python and
//! FFmpeg-subprocess approaches introduce unacceptable host↔device copy overhead,
//! GIL contention, and unbounded VRAM growth.
//!
//! ## Pipeline
//!
//! ```text
//! ┌──────────┐  ch(4)  ┌────────────┐  ch(2)  ┌───────────┐  ch(4)  ┌──────────┐
//! │ Decoder  │────────►│ Preprocess │────────►│ Inference │────────►│ Encoder  │
//! │ (NVDEC)  │         │(CUDA kern) │         │(TensorRT) │         │ (NVENC)  │
//! │ NV12 GPU │         │NV12→RGB F32│         │RGB F32/16 │         │NV12 GPU  │
//! └──────────┘         └────────────┘         └───────────┘         └──────────┘
//!
//! GPU-resident frame transport. No host copies in steady state.
//! ```
//!
//! Four concurrent stages connected by bounded `tokio::sync::mpsc` channels
//! (capacities 4, 2, 4). Backpressure flows upstream: when the encoder stalls,
//! channel saturation propagates back through inference → preprocess → decode,
//! bounding total in-flight VRAM to `sum(channel_capacities) × frame_size`.
//!
//! ## Modules
//!
//! - [`core`] — GPU contract types ([`core::types::GpuTexture`], [`core::types::FrameEnvelope`],
//!   [`core::types::PixelFormat`]), shared GPU context and bucketed buffer pool
//!   ([`core::context::GpuContext`]), CUDA preprocessing kernels compiled via NVRTC
//!   ([`core::kernels::PreprocessKernels`]), and the [`core::backend::UpscaleBackend`] trait
//!   that abstracts inference backends.
//!
//! - [`backends`] — Concrete inference backend implementations. Currently provides
//!   [`backends::tensorrt::TensorRtBackend`] which wraps an ONNX Runtime session with
//!   the TensorRT execution provider, using IO binding for device-resident tensor handoff.
//!   Includes [`backends::tensorrt::BatchConfig`] for optional micro-batch collection with
//!   a latency deadline.
//!
//! - [`codecs`] — NVIDIA hardware codec wrappers. [`codecs::nvdec::NvDecoder`] drives
//!   NVDEC via the Video Codec SDK parser/decoder API. [`codecs::nvenc::NvEncoder`]
//!   drives NVENC for H.264/HEVC encoding. [`codecs::sys`] contains raw FFI bindings
//!   for `nvcuvid`, `nvEncodeAPI`, and CUDA driver types.
//!
//! - [`engine`] — Pipeline orchestration. [`engine::pipeline::UpscalePipeline`] spawns
//!   four async stages (decode, preprocess, inference, encode) connected by bounded
//!   `tokio::sync::mpsc` channels. [`engine::pipeline::PipelineMetrics`] provides
//!   lock-free atomic counters for per-stage frame counts and latency accumulation.
//!
//! - [`io`] — Container and raw bitstream I/O. FFmpeg-based demuxer/muxer for MP4/MKV/MOV
//!   containers, raw `BitstreamSource`/`BitstreamSink` for Annex B files, and container
//!   probing for codec/resolution/framerate detection.
//!
//! - [`error`] — Typed error hierarchy ([`error::EngineError`]) with stable numeric error
//!   codes for every failure variant. Provides the crate-wide [`error::Result`] alias.
//!
//! ## Design principles
//!
//! - **GPU-resident frame transport**: Frame data ([`core::types::GpuTexture`]) is allocated
//!   in device memory and stays there across all four pipeline stages. No `cudaMemcpy` in
//!   steady state. Frames are wrapped in `Arc<CudaSlice<u8>>` for reference-counted,
//!   Send + Sync sharing across stages.
//! - **Bounded backpressure**: Channel capacities between stages (configurable via
//!   [`engine::pipeline::PipelineConfig`]) prevent fast producers from overwhelming slow
//!   consumers or exhausting VRAM. Queue depths are tracked via lock-free atomic counters
//!   ([`core::context::QueueDepthTracker`]).
//! - **Bucketed buffer pool**: After warm-up, all device memory comes from a recycling
//!   pool keyed by buffer size — zero CUDA driver allocations in steady state. Pool
//!   statistics (hits, misses, recycles, overflows) are tracked atomically.
//! - **RAII resource management**: All GPU resources (device memory, CUDA events,
//!   decoder/encoder handles) are cleaned up via Rust ownership and `Drop` semantics.
//!
//! ## Safety
//!
//! This crate contains `unsafe` code for FFI interop with three C APIs:
//!
//! - **CUDA Driver API** (via cudarc): Device memory allocation, stream synchronization,
//!   kernel launch. cudarc provides safe wrappers; raw driver calls are used only where
//!   cudarc's API is insufficient (e.g., extracting the underlying `CUcontext` handle).
//! - **NVIDIA Video Codec SDK**: NVDEC (`nvcuvid`) and NVENC (`nvEncodeAPI`) require
//!   raw pointer manipulation and callback-driven APIs. The [`codecs::sys`] module
//!   contains the FFI declarations; safe wrappers in [`codecs::nvdec`] and
//!   [`codecs::nvenc`] enforce lifetime and ownership invariants.
//! - **FFmpeg** (`libavcodec`, `libavformat`): Container demux/mux uses `ffmpeg-sys-next`
//!   bindings. Resource cleanup is handled via RAII wrapper types in the [`io`] module.

pub mod backends;
pub mod codecs;
pub mod core;
pub mod debug_alloc;
pub mod engine;
pub mod error;
pub mod io;
pub mod probe_ort;

#[cfg(feature = "debug-alloc")]
#[global_allocator]
static ALLOC: crate::debug_alloc::TrackingAllocator = crate::debug_alloc::TrackingAllocator;
