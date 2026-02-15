//! Typed error hierarchy for the engine.
//!
//! Uses `thiserror` for library-grade errors.  Application code should wrap
//! these in `anyhow::Result` at call sites.
//!
//! # Error codes (Phase 9)
//!
//! Each variant maps to a stable integer code via [`EngineError::error_code`]
//! for structured telemetry without string parsing.

use crate::core::types::PixelFormat;

/// All errors originating from the VideoForge engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    // ── CUDA ──────────────────────────────────────────────────────────
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error("CUDA kernel compilation error: {0}")]
    NvrtcCompile(#[from] cudarc::nvrtc::CompileError),

    // ── Inference ────────────────────────────────────────────────────
    #[error("ORT inference error: {0}")]
    Inference(#[from] ort::Error),

    #[error("Model metadata error: {0}")]
    ModelMetadata(String),

    #[error("Backend not initialized — call initialize() first")]
    NotInitialized,

    // ── Codecs ────────────────────────────────────────────────────────
    #[error("NVDEC decode error: {0}")]
    Decode(String),

    #[error("NVENC encode error: {0}")]
    Encode(String),

    #[error("Demux error: {0}")]
    Demux(String),

    #[error("Mux error: {0}")]
    Mux(String),

    #[error("Bitstream filter error: {0}")]
    BitstreamFilter(String),

    #[error("Probe error: {0}")]
    Probe(String),

    // ── Pipeline ─────────────────────────────────────────────────────
    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Pipeline channel closed unexpectedly")]
    ChannelClosed,

    #[error("Pipeline shutdown signal received")]
    Shutdown,

    // ── Type contracts ───────────────────────────────────────────────
    #[error("Pixel format mismatch: expected {expected:?}, got {actual:?}")]
    FormatMismatch {
        expected: PixelFormat,
        actual: PixelFormat,
    },

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Buffer too small: need {need} bytes, have {have}")]
    BufferTooSmall { need: usize, have: usize },

    // ── Audit invariants ─────────────────────────────────────────────
    #[error("Invariant violation: {0}")]
    InvariantViolation(String),

    // ── Production hardening (Phase 9) ───────────────────────────────
    #[error("Panic recovered in {stage}: {message}")]
    PanicRecovered {
        stage: &'static str,
        message: String,
    },

    #[error("VRAM limit exceeded: current {current_mb} MiB > cap {limit_mb} MiB")]
    VramLimitExceeded { current_mb: usize, limit_mb: usize },

    #[error("Backpressure timeout: {stage} blocked for {elapsed_ms} ms")]
    BackpressureTimeout {
        stage: &'static str,
        elapsed_ms: u64,
    },

    #[error("Drop order violation: {0}")]
    DropOrderViolation(String),
}

impl EngineError {
    /// Stable integer error code for structured telemetry.
    ///
    /// Codes are grouped by category:
    /// - 1xx: CUDA/driver
    /// - 2xx: Inference
    /// - 3xx: Codecs
    /// - 4xx: Pipeline
    /// - 5xx: Type contracts
    /// - 6xx: Audit/invariant
    /// - 7xx: Production hardening
    pub fn error_code(&self) -> u32 {
        match self {
            Self::Cuda(_) => 100,
            Self::NvrtcCompile(_) => 101,
            Self::Inference(_) => 200,
            Self::ModelMetadata(_) => 201,
            Self::NotInitialized => 202,
            Self::Decode(_) => 300,
            Self::Encode(_) => 301,
            Self::Demux(_) => 302,
            Self::Mux(_) => 303,
            Self::BitstreamFilter(_) => 304,
            Self::Probe(_) => 305,
            Self::ChannelClosed => 400,
            Self::Shutdown => 401,
            Self::Pipeline(_) => 402,
            Self::FormatMismatch { .. } => 500,
            Self::DimensionMismatch(_) => 501,
            Self::BufferTooSmall { .. } => 502,
            Self::InvariantViolation(_) => 600,
            Self::PanicRecovered { .. } => 700,
            Self::VramLimitExceeded { .. } => 701,
            Self::BackpressureTimeout { .. } => 702,
            Self::DropOrderViolation(_) => 703,
        }
    }

    /// Whether this error is recoverable (pipeline can continue after logging).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::BackpressureTimeout { .. } | Self::PanicRecovered { .. }
        )
    }
}

/// Convenience alias used throughout the engine crate.
pub type Result<T> = std::result::Result<T, EngineError>;
