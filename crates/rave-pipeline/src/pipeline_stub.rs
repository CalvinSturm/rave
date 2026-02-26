#![allow(missing_docs)]
//! Stub pipeline API for docs.rs / no-CUDA builds.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use rave_core::backend::UpscaleBackend;
use rave_core::codec_traits::{FrameDecoder, FrameEncoder};
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};

use crate::stage_graph::{PipelineReport, ProfilePreset, RunContract, StageGraph};

#[derive(Debug)]
pub struct PipelineMetrics {
    pub frames_decoded: AtomicU64,
    pub frames_preprocessed: AtomicU64,
    pub frames_inferred: AtomicU64,
    pub frames_encoded: AtomicU64,
    pub decode_total_us: AtomicU64,
    pub preprocess_total_us: AtomicU64,
    pub inference_total_us: AtomicU64,
    pub postprocess_total_us: AtomicU64,
    pub encode_total_us: AtomicU64,
}

impl PipelineMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            frames_decoded: AtomicU64::new(0),
            frames_preprocessed: AtomicU64::new(0),
            frames_inferred: AtomicU64::new(0),
            frames_encoded: AtomicU64::new(0),
            decode_total_us: AtomicU64::new(0),
            preprocess_total_us: AtomicU64::new(0),
            inference_total_us: AtomicU64::new(0),
            postprocess_total_us: AtomicU64::new(0),
            encode_total_us: AtomicU64::new(0),
        })
    }

    pub fn validate(&self) -> bool {
        let d = self.frames_decoded.load(Ordering::Acquire);
        let p = self.frames_preprocessed.load(Ordering::Acquire);
        let i = self.frames_inferred.load(Ordering::Acquire);
        let e = self.frames_encoded.load(Ordering::Acquire);
        d >= p && p >= i && i >= e
    }

    pub fn report(&self) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DeterminismPolicy {
    #[default]
    BestEffort,
    RequireHash,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeterminismSkipReason {
    FeatureDisabled,
    UnsupportedFormat,
    DebugReadbackUnavailable,
    BackendNoReadback,
    ExplicitlyDisabled,
    Unknown,
}

impl DeterminismSkipReason {
    pub fn code(self) -> &'static str {
        match self {
            Self::FeatureDisabled => "feature_disabled",
            Self::UnsupportedFormat => "unsupported_format",
            Self::DebugReadbackUnavailable => "debug_readback_unavailable",
            Self::BackendNoReadback => "backend_no_readback",
            Self::ExplicitlyDisabled => "explicitly_disabled",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DeterminismObserved {
    pub hash_requested: bool,
    pub hash_available: bool,
    pub skip_reason: Option<DeterminismSkipReason>,
}

pub fn enforce_determinism_policy(
    policy: DeterminismPolicy,
    observed: DeterminismObserved,
) -> Result<()> {
    if !observed.hash_requested || observed.hash_available {
        return Ok(());
    }
    match policy {
        DeterminismPolicy::BestEffort => Ok(()),
        DeterminismPolicy::RequireHash => Err(EngineError::InvariantViolation(format!(
            "determinism hash required in production_strict, but was skipped: {}",
            observed
                .skip_reason
                .unwrap_or(DeterminismSkipReason::Unknown)
                .code()
        ))),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum StubModelPrecision {
    #[default]
    F32,
    F16,
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub decoded_capacity: usize,
    pub preprocessed_capacity: usize,
    pub upscaled_capacity: usize,
    pub encoder_nv12_pitch: usize,
    pub model_precision: StubModelPrecision,
    pub enable_profiler: bool,
    pub strict_no_host_copies: bool,
    pub strict_invariants: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            decoded_capacity: 4,
            preprocessed_capacity: 2,
            upscaled_capacity: 4,
            encoder_nv12_pitch: 0,
            model_precision: StubModelPrecision::F32,
            enable_profiler: true,
            strict_no_host_copies: false,
            strict_invariants: false,
        }
    }
}

pub struct UpscalePipeline {
    cancel: CancellationToken,
    metrics: Arc<PipelineMetrics>,
}

impl UpscalePipeline {
    pub fn new(_ctx: Arc<GpuContext>, _kernels: Arc<()>, _config: PipelineConfig) -> Self {
        Self {
            cancel: CancellationToken::new(),
            metrics: PipelineMetrics::new(),
        }
    }

    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    pub fn metrics(&self) -> Arc<PipelineMetrics> {
        self.metrics.clone()
    }

    pub async fn run<D, B, E>(&self, _decoder: D, _backend: Arc<B>, _encoder: E) -> Result<()>
    where
        D: FrameDecoder,
        B: UpscaleBackend + 'static,
        E: FrameEncoder,
    {
        Err(feature_disabled_err())
    }

    pub async fn run_graph(
        &self,
        _input: &Path,
        _output: &Path,
        _graph: StageGraph,
        _profile: ProfilePreset,
        _contract: RunContract,
    ) -> Result<PipelineReport> {
        Err(feature_disabled_err())
    }
}

#[derive(Debug)]
pub struct StressTestReport {
    pub total_frames: u64,
    pub elapsed: Duration,
    pub avg_fps: f64,
    pub avg_latency_ms: f64,
    pub peak_vram_bytes: usize,
    pub final_vram_bytes: usize,
    pub vram_before_bytes: usize,
    pub frames_decoded: u64,
    pub frames_encoded: u64,
    pub pool_hit_rate_pct: f64,
}

#[derive(Debug)]
pub struct AuditReport {
    pub host_alloc_check: AuditResult,
    pub vram_leak_check: AuditResult,
    pub pool_hit_rate_check: AuditResult,
    pub stream_overlap_check: AuditResult,
}

#[derive(Debug)]
pub enum AuditResult {
    Pass(String),
    Fail(String),
}

impl AuditResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass(_))
    }
}

impl AuditReport {
    pub fn all_pass(&self) -> bool {
        self.host_alloc_check.is_pass()
            && self.vram_leak_check.is_pass()
            && self.pool_hit_rate_check.is_pass()
            && self.stream_overlap_check.is_pass()
    }
}

pub struct AuditSuite;

impl AuditSuite {
    pub async fn run_all<B>(
        _ctx: Arc<GpuContext>,
        _kernels: Arc<()>,
        _backend: Arc<B>,
        _config: PipelineConfig,
    ) -> Result<AuditReport>
    where
        B: UpscaleBackend + 'static,
    {
        Err(feature_disabled_err())
    }
}

fn feature_disabled_err() -> EngineError {
    EngineError::Pipeline(
        "rave-pipeline built without `cuda-pipeline`; runtime pipeline APIs are unavailable".into(),
    )
}
