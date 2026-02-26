#![allow(missing_docs)]
//! Stub TensorRT backend for docs.rs / no-runtime builds.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use rave_core::backend::{ModelMetadata, UpscaleBackend};
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::types::GpuTexture;

/// TensorRT precision policy — controls EP optimization flags.
#[derive(Clone, Debug, Default)]
pub enum PrecisionPolicy {
    /// FP32 only — maximum accuracy, baseline performance.
    Fp32,
    /// FP16 mixed precision — 2× throughput on Tensor Cores.
    #[default]
    Fp16,
    /// INT8 quantized with calibration table — 4× throughput.
    Int8 { calibration_table: PathBuf },
}

/// Batch inference configuration.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub max_batch: usize,
    pub latency_deadline_us: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch: 1,
            latency_deadline_us: 8_000,
        }
    }
}

/// Validate a [`BatchConfig`], returning an error if `max_batch > 1`.
pub fn validate_batch_config(cfg: &BatchConfig) -> Result<()> {
    if cfg.max_batch > 1 {
        return Err(EngineError::InvariantViolation(
            "micro-batching is not implemented; max_batch must be 1 (set max_batch=1)".into(),
        ));
    }
    Ok(())
}

/// Atomic counters for inference stage observability.
#[derive(Debug)]
pub struct InferenceMetrics {
    pub frames_inferred: AtomicU64,
    pub total_inference_us: AtomicU64,
    pub peak_inference_us: AtomicU64,
}

impl InferenceMetrics {
    pub const fn new() -> Self {
        Self {
            frames_inferred: AtomicU64::new(0),
            total_inference_us: AtomicU64::new(0),
            peak_inference_us: AtomicU64::new(0),
        }
    }

    pub fn record(&self, elapsed_us: u64) {
        self.frames_inferred.fetch_add(1, Ordering::Relaxed);
        self.total_inference_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        self.peak_inference_us
            .fetch_max(elapsed_us, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> InferenceMetricsSnapshot {
        let frames = self.frames_inferred.load(Ordering::Relaxed);
        let total = self.total_inference_us.load(Ordering::Relaxed);
        let peak = self.peak_inference_us.load(Ordering::Relaxed);
        InferenceMetricsSnapshot {
            frames_inferred: frames,
            avg_inference_us: if frames > 0 { total / frames } else { 0 },
            peak_inference_us: peak,
        }
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of inference metrics for reporting.
#[derive(Clone, Debug)]
pub struct InferenceMetricsSnapshot {
    pub frames_inferred: u64,
    pub avg_inference_us: u64,
    pub peak_inference_us: u64,
}

/// A point-in-time snapshot of [`RingMetrics`] counters.
#[derive(Debug, Clone, Copy)]
pub struct RingMetricsSnapshot {
    pub reuse: u64,
    pub contention: u64,
    pub first_use: u64,
}

/// Atomic counters for output ring buffer activity.
#[derive(Debug)]
pub struct RingMetrics {
    pub slot_reuse_count: AtomicU64,
    pub slot_contention_events: AtomicU64,
    pub slot_first_use_count: AtomicU64,
}

impl RingMetrics {
    pub const fn new() -> Self {
        Self {
            slot_reuse_count: AtomicU64::new(0),
            slot_contention_events: AtomicU64::new(0),
            slot_first_use_count: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> RingMetricsSnapshot {
        RingMetricsSnapshot {
            reuse: self.slot_reuse_count.load(Ordering::Relaxed),
            contention: self.slot_contention_events.load(Ordering::Relaxed),
            first_use: self.slot_first_use_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for RingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size ring of pre-allocated device buffers for inference output.
pub struct OutputRing {
    pub slot_bytes: usize,
    pub alloc_dims: (u32, u32),
    pub metrics: RingMetrics,
}

impl OutputRing {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _ctx: &GpuContext,
        _in_w: u32,
        _in_h: u32,
        _scale: u32,
        _count: usize,
        _min_slots: usize,
    ) -> Result<Self> {
        Err(runtime_disabled_err())
    }
}

/// Stub TensorRT/CUDA ORT inference backend.
pub struct TensorRtBackend {
    /// Atomic inference latency and frame count metrics.
    pub inference_metrics: InferenceMetrics,
    /// Precision policy used when building the TensorRT EP session.
    pub precision_policy: PrecisionPolicy,
    /// Batch configuration.
    pub batch_config: BatchConfig,
    selected_provider: Option<String>,
}

impl TensorRtBackend {
    pub fn new(
        model_path: PathBuf,
        ctx: Arc<GpuContext>,
        device_id: i32,
        ring_size: usize,
        downstream_capacity: usize,
    ) -> Self {
        Self::with_precision(
            model_path,
            ctx,
            device_id,
            ring_size,
            downstream_capacity,
            PrecisionPolicy::default(),
            BatchConfig::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_precision(
        _model_path: PathBuf,
        _ctx: Arc<GpuContext>,
        _device_id: i32,
        _ring_size: usize,
        _downstream_capacity: usize,
        precision_policy: PrecisionPolicy,
        batch_config: BatchConfig,
    ) -> Self {
        Self {
            inference_metrics: InferenceMetrics::new(),
            precision_policy,
            batch_config,
            selected_provider: None,
        }
    }

    pub async fn ring_metrics(&self) -> Option<RingMetricsSnapshot> {
        None
    }

    pub fn selected_provider(&self) -> Option<&str> {
        self.selected_provider.as_deref()
    }
}

#[async_trait]
impl UpscaleBackend for TensorRtBackend {
    async fn initialize(&self) -> Result<()> {
        Err(runtime_disabled_err())
    }

    async fn process(&self, _input: GpuTexture) -> Result<GpuTexture> {
        Err(runtime_disabled_err())
    }

    async fn shutdown(&self) -> Result<()> {
        Err(runtime_disabled_err())
    }

    fn metadata(&self) -> Result<&ModelMetadata> {
        Err(runtime_disabled_err())
    }
}

fn runtime_disabled_err() -> EngineError {
    EngineError::Inference(
        "rave-tensorrt built without `tensorrt-runtime`; TensorRT backend is unavailable".into(),
    )
}
