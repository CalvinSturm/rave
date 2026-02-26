use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use rave_core::error::{EngineError, Result};

/// JSON schema version for [`StageGraph`].  Bump on breaking schema changes.
pub const GRAPH_SCHEMA_VERSION: u32 = 1;

/// Unique identifier for a processing stage within a [`StageGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StageId(pub u32);

/// The kind of processing a stage performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    /// Super-resolution upscale via TensorRT.
    Enhance,
}

/// Named configuration preset that controls runtime contracts and validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfilePreset {
    /// Development: relaxed contracts, warnings tolerated, host copies allowed.
    Dev,
    /// Production: strict no-host-copies contract, deterministic output, audit warnings fail.
    ProductionStrict,
    /// Benchmark: same as Dev but disables determinism checks for max throughput.
    Benchmark,
}

impl ProfilePreset {
    /// Returns `true` if this preset enforces the strict no-host-copies invariant.
    pub fn strict_no_host_copies(self) -> bool {
        matches!(self, Self::ProductionStrict)
    }

    /// Returns `true` if this preset requires deterministic output validation.
    pub fn deterministic_contract(self) -> bool {
        matches!(self, Self::ProductionStrict)
    }

    /// Returns `true` if audit warnings should be promoted to hard errors.
    pub fn fail_on_audit_warn(self) -> bool {
        matches!(self, Self::ProductionStrict)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunContract {
    /// Preferred CUDA device ordinal used in report metadata.
    pub requested_device: u32,
    /// Enable deterministic output checks and checkpoint comparison.
    pub deterministic_output: bool,
    /// Number of canonical stage frames to hash for determinism checks.
    pub determinism_hash_frames: usize,
    /// Treat audit warnings as hard errors.
    pub fail_on_audit_warn: bool,
}

impl Default for RunContract {
    fn default() -> Self {
        Self::for_profile(ProfilePreset::Dev)
    }
}

impl RunContract {
    /// Construct a [`RunContract`] with defaults appropriate for `profile`.
    pub fn for_profile(profile: ProfilePreset) -> Self {
        Self {
            requested_device: 0,
            deterministic_output: profile.deterministic_contract(),
            determinism_hash_frames: 0,
            fail_on_audit_warn: profile.fail_on_audit_warn(),
        }
    }
}

/// Floating-point precision selection for the TensorRT inference stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionPolicyConfig {
    /// 32-bit single-precision float (higher quality, more VRAM).
    Fp32,
    /// 16-bit half-precision float (default — faster inference, less VRAM).
    #[default]
    Fp16,
}

impl PrecisionPolicyConfig {
    #[cfg(feature = "nvidia-run-graph")]
    /// Map to the CUDA kernel precision enum.
    pub fn as_model_precision(self) -> rave_cuda::kernels::ModelPrecision {
        match self {
            Self::Fp32 => rave_cuda::kernels::ModelPrecision::F32,
            Self::Fp16 => rave_cuda::kernels::ModelPrecision::F16,
        }
    }

    #[cfg(feature = "nvidia-run-graph")]
    /// Map to the TensorRT precision policy enum.
    pub fn as_tensorrt_precision(self) -> rave_tensorrt::tensorrt::PrecisionPolicy {
        match self {
            Self::Fp32 => rave_tensorrt::tensorrt::PrecisionPolicy::Fp32,
            Self::Fp16 => rave_tensorrt::tensorrt::PrecisionPolicy::Fp16,
        }
    }
}

/// Micro-batching configuration for the inference stage.
///
/// **Note:** `max_batch > 1` is not yet implemented and will fail validation.
/// This struct exists for future use.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum frames per inference batch.  Must be `1` (batching not yet implemented).
    pub max_batch: usize,
    /// Maximum acceptable inference latency in microseconds before the batcher flushes.
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

impl BatchConfig {
    /// Convert to the TensorRT-level batch config.
    #[cfg(feature = "nvidia-run-graph")]
    pub fn to_tensorrt(&self) -> rave_tensorrt::tensorrt::BatchConfig {
        rave_tensorrt::tensorrt::BatchConfig {
            max_batch: self.max_batch,
            latency_deadline_us: self.latency_deadline_us,
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

/// Configuration for a [`StageKind::Enhance`] stage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnhanceConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Floating-point precision policy for this stage.
    #[serde(default)]
    pub precision_policy: PrecisionPolicyConfig,
    /// Micro-batching configuration (must have `max_batch = 1`).
    #[serde(default)]
    pub batch_config: BatchConfig,
    /// Expected upscale factor reported by the model (default `2`).
    #[serde(default = "default_scale")]
    pub scale: u32,
}

fn default_scale() -> u32 {
    2
}

/// Per-stage configuration tagged by [`StageKind`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageConfig {
    /// Super-resolution upscale stage.
    Enhance { id: StageId, config: EnhanceConfig },
}

impl StageConfig {
    /// Return the unique identifier of this stage.
    pub fn id(&self) -> StageId {
        match self {
            Self::Enhance { id, .. } => *id,
        }
    }

    /// Return the [`StageKind`] discriminant of this stage.
    pub fn kind(&self) -> StageKind {
        match self {
            Self::Enhance { .. } => StageKind::Enhance,
        }
    }

    /// Return `Some(&EnhanceConfig)` if this is an [`Enhance`](StageConfig::Enhance) stage.
    pub fn as_enhance(&self) -> Option<&EnhanceConfig> {
        match self {
            Self::Enhance { config, .. } => Some(config),
        }
    }
}

/// Ordered list of processing stages describing the full pipeline configuration.
///
/// Serialises to/from JSON with a `graph_schema_version` field for
/// forward-compatibility gating.  Use [`StageGraph::from_json_str`] or
/// [`StageGraph::from_json_file`] for safe deserialization with version checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageGraph {
    /// Must equal [`GRAPH_SCHEMA_VERSION`]; checked on deserialization and validation.
    pub graph_schema_version: u32,
    /// Ordered list of stage configurations.
    pub stages: Vec<StageConfig>,
}

impl Default for StageGraph {
    fn default() -> Self {
        Self {
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            stages: Vec::new(),
        }
    }
}

impl StageGraph {
    /// Deserialize a [`StageGraph`] from a JSON string.
    ///
    /// Returns an error if the JSON is malformed or if `graph_schema_version`
    /// does not match [`GRAPH_SCHEMA_VERSION`].
    pub fn from_json_str(data: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(data).map_err(|err| {
            EngineError::InvariantViolation(format!("Invalid stage graph JSON: {err}"))
        })?;

        let Some(version_value) = value.get("graph_schema_version") else {
            return Err(EngineError::InvariantViolation(format!(
                "Graph schema mismatch: expected {}, got missing",
                GRAPH_SCHEMA_VERSION
            )));
        };
        let Some(version) = version_value.as_u64() else {
            return Err(EngineError::InvariantViolation(format!(
                "Graph schema mismatch: expected {}, got non-integer",
                GRAPH_SCHEMA_VERSION
            )));
        };
        if version != GRAPH_SCHEMA_VERSION as u64 {
            return Err(EngineError::InvariantViolation(format!(
                "Graph schema mismatch: expected {}, got {}",
                GRAPH_SCHEMA_VERSION, version
            )));
        }

        serde_json::from_value(value).map_err(|err| {
            EngineError::InvariantViolation(format!("Invalid stage graph JSON: {err}"))
        })
    }

    /// Deserialize a [`StageGraph`] from a JSON file on disk.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(|err| {
            EngineError::InvariantViolation(format!(
                "Failed to read stage graph from {}: {err}",
                path.display()
            ))
        })?;
        Self::from_json_str(&data)
    }

    /// Validate graph invariants: schema version, stage ID uniqueness, required
    /// Enhance stage presence, and per-stage parameter constraints.
    pub fn validate(&self) -> Result<()> {
        if self.graph_schema_version != GRAPH_SCHEMA_VERSION {
            return Err(EngineError::InvariantViolation(format!(
                "Graph schema mismatch: expected {}, got {}",
                GRAPH_SCHEMA_VERSION, self.graph_schema_version
            )));
        }
        if self.stages.is_empty() {
            return Err(EngineError::InvariantViolation(
                "StageGraph validation failed: at least one stage is required".into(),
            ));
        }

        let mut ids = BTreeSet::new();
        let mut enhance_count = 0usize;
        for stage in &self.stages {
            if !ids.insert(stage.id()) {
                return Err(EngineError::InvariantViolation(format!(
                    "StageGraph validation failed: duplicate stage id {}",
                    stage.id().0
                )));
            }

            match stage {
                StageConfig::Enhance { id, config } => {
                    enhance_count += 1;
                    if config.model_path.as_os_str().is_empty() {
                        return Err(EngineError::InvariantViolation(format!(
                            "StageGraph validation failed: stage {} {:?} is missing model_path",
                            id.0,
                            StageKind::Enhance
                        )));
                    }
                    validate_batch_config(&config.batch_config)?;
                }
            }
        }

        if enhance_count == 0 {
            return Err(EngineError::InvariantViolation(
                "StageGraph validation failed: missing enhance stage".into(),
            ));
        }
        if enhance_count > 1 {
            return Err(EngineError::InvariantViolation(
                "StageGraph validation failed: multiple enhance stages are not supported in MVP"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Return the [`EnhanceConfig`] for the single required Enhance stage, if present.
    pub fn single_enhance_config(&self) -> Option<&EnhanceConfig> {
        self.stages.iter().find_map(StageConfig::as_enhance)
    }
}

/// Severity level of a pipeline audit finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditLevel {
    /// Invariant satisfied — no action required.
    Pass,
    /// Potential issue detected (promoted to error under [`ProfilePreset::ProductionStrict`]).
    Warn,
    /// Hard failure — pipeline should be aborted.
    Fail,
}

/// A single audit finding emitted by the pipeline's invariant-checking layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditItem {
    /// Severity of this finding.
    pub level: AuditLevel,
    /// Short machine-readable audit code (e.g. `"host_copy_detected"`).
    pub code: String,
    /// The stage that triggered this finding, if applicable.
    pub stage_id: Option<StageId>,
    /// Human-readable description of the finding.
    pub message: String,
}

/// Per-stage wall-clock timing averages for one pipeline run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageTimingReport {
    /// Average decode stage latency in microseconds.
    pub decode_us: u64,
    /// Average preprocess stage latency in microseconds.
    pub preprocess_us: u64,
    /// Average inference stage latency in microseconds.
    pub infer_us: u64,
    /// Average postprocess stage latency in microseconds.
    pub postprocess_us: u64,
    /// Average encode stage latency in microseconds.
    pub encode_us: u64,
}

/// Structured summary report produced at the end of a pipeline run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineReport {
    /// CUDA device ordinal that was used.
    pub selected_device: u32,
    /// ORT execution provider name (e.g. `"TensorrtExecutionProvider"`).
    pub provider: String,
    /// Model identifier string from the ONNX graph.
    pub model_name: String,
    /// Spatial upscale factor reported by the model.
    pub model_scale: u32,
    /// Output frame width in pixels.
    pub output_width: u32,
    /// Output frame height in pixels.
    pub output_height: u32,
    /// Total frames decoded.
    pub frames_decoded: u64,
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Per-stage timing averages.
    pub stage_timing: StageTimingReport,
    /// FNV-1a checkpoint hashes for determinism validation.
    pub stage_checksums: Vec<String>,
    /// Current VRAM usage at pipeline shutdown in bytes.
    pub vram_current_bytes: usize,
    /// Peak VRAM usage observed during the run in bytes.
    pub vram_peak_bytes: usize,
    /// Audit findings from the invariant-checking layer.
    pub audit: Vec<AuditItem>,
}

/// Compute a deterministic FNV-1a 64-bit hash of `bytes` for checkpoint comparisons.
pub fn hash_checkpoint_bytes(bytes: &[u8]) -> String {
    // Deterministic FNV-1a 64-bit hash for lightweight checkpointing.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enhance_stage(id: u32) -> StageConfig {
        StageConfig::Enhance {
            id: StageId(id),
            config: EnhanceConfig {
                model_path: PathBuf::from("model.onnx"),
                precision_policy: PrecisionPolicyConfig::Fp16,
                batch_config: BatchConfig::default(),
                scale: 2,
            },
        }
    }

    #[test]
    fn graph_validation_requires_enhance() {
        let graph = StageGraph {
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            stages: vec![],
        };
        let err = graph
            .validate()
            .expect_err("must fail without enhance stage");
        assert!(
            err.to_string().contains("at least one stage is required")
                || err.to_string().contains("missing enhance stage")
        );
    }

    #[test]
    fn graph_validation_rejects_multiple_enhance_stages() {
        let graph = StageGraph {
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            stages: vec![enhance_stage(1), enhance_stage(2)],
        };
        let err = graph
            .validate()
            .expect_err("must fail with multiple enhance stages");
        assert!(err.to_string().contains("multiple enhance stages"));
    }

    #[test]
    fn graph_validation_accepts_single_enhance_chain() {
        let graph = StageGraph {
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            stages: vec![enhance_stage(1)],
        };
        graph.validate().expect("graph should be valid");
    }

    #[test]
    fn graph_validation_rejects_micro_batching() {
        let mut stage = enhance_stage(1);
        let StageConfig::Enhance { config, .. } = &mut stage else {
            panic!("expected enhance stage");
        };
        config.batch_config.max_batch = 2;

        let graph = StageGraph {
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            stages: vec![stage],
        };
        let err = graph
            .validate()
            .expect_err("micro-batching must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("max_batch"));
        assert!(msg.contains("not implemented"));
    }

    #[test]
    fn graph_validation_rejects_schema_mismatch() {
        let graph = StageGraph {
            graph_schema_version: GRAPH_SCHEMA_VERSION + 1,
            stages: vec![enhance_stage(1)],
        };
        let err = graph.validate().expect_err("schema mismatch must fail");
        assert!(err.to_string().contains("Graph schema mismatch"));
    }

    #[test]
    fn checkpoint_hash_is_deterministic() {
        let a = hash_checkpoint_bytes(b"abc123");
        let b = hash_checkpoint_bytes(b"abc123");
        let c = hash_checkpoint_bytes(b"abc124");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn from_json_rejects_missing_schema_version() {
        let raw = r#"{"stages":[]}"#;
        let err = StageGraph::from_json_str(raw).expect_err("missing schema must fail");
        assert!(err.to_string().contains("Graph schema mismatch"));
    }

    #[test]
    fn from_json_rejects_schema_version_mismatch() {
        let raw = format!(
            "{{\"graph_schema_version\":{},\"stages\":[]}}",
            GRAPH_SCHEMA_VERSION + 1
        );
        let err = StageGraph::from_json_str(&raw).expect_err("mismatch schema must fail");
        assert!(err.to_string().contains("Graph schema mismatch"));
    }
}
