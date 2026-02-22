use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use rave_core::error::{EngineError, Result};
use rave_cuda::kernels::ModelPrecision;
use rave_tensorrt::tensorrt::{
    PrecisionPolicy, validate_batch_config as validate_tensorrt_batch_config,
};

pub const GRAPH_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StageId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    Enhance,
    FaceBlur,
    FaceSwapAndEnhance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfilePreset {
    Dev,
    ProductionStrict,
    Benchmark,
}

impl ProfilePreset {
    pub fn strict_no_host_copies(self) -> bool {
        matches!(self, Self::ProductionStrict)
    }

    pub fn deterministic_contract(self) -> bool {
        matches!(self, Self::ProductionStrict)
    }

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
    pub fn for_profile(profile: ProfilePreset) -> Self {
        Self {
            requested_device: 0,
            deterministic_output: profile.deterministic_contract(),
            determinism_hash_frames: 0,
            fail_on_audit_warn: profile.fail_on_audit_warn(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionPolicyConfig {
    Fp32,
    #[default]
    Fp16,
}

impl PrecisionPolicyConfig {
    pub fn as_model_precision(self) -> ModelPrecision {
        match self {
            Self::Fp32 => ModelPrecision::F32,
            Self::Fp16 => ModelPrecision::F16,
        }
    }

    pub fn as_tensorrt_precision(self) -> PrecisionPolicy {
        match self {
            Self::Fp32 => PrecisionPolicy::Fp32,
            Self::Fp16 => PrecisionPolicy::Fp16,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

impl BatchConfig {
    pub fn to_tensorrt(&self) -> rave_tensorrt::tensorrt::BatchConfig {
        rave_tensorrt::tensorrt::BatchConfig {
            max_batch: self.max_batch,
            latency_deadline_us: self.latency_deadline_us,
        }
    }
}

pub fn validate_batch_config(cfg: &BatchConfig) -> Result<()> {
    let backend_cfg = cfg.to_tensorrt();
    validate_tensorrt_batch_config(&backend_cfg)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlurMode {
    #[default]
    WholeFrameGaussian,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnhanceConfig {
    pub model_path: PathBuf,
    #[serde(default)]
    pub precision_policy: PrecisionPolicyConfig,
    #[serde(default)]
    pub batch_config: BatchConfig,
    #[serde(default = "default_scale")]
    pub scale: u32,
}

fn default_scale() -> u32 {
    2
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlurConfig {
    pub sigma: f32,
    pub iterations: u32,
    #[serde(default)]
    pub mode: BlurMode,
    #[serde(default)]
    pub regions: Option<Vec<Rect>>,
}

impl Default for BlurConfig {
    fn default() -> Self {
        Self {
            sigma: 1.2,
            iterations: 1,
            mode: BlurMode::WholeFrameGaussian,
            regions: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwapConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl Default for SwapConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageConfig {
    Enhance { id: StageId, config: EnhanceConfig },
    FaceBlur { id: StageId, config: BlurConfig },
    FaceSwapAndEnhance { id: StageId, config: SwapConfig },
}

impl StageConfig {
    pub fn id(&self) -> StageId {
        match self {
            Self::Enhance { id, .. }
            | Self::FaceBlur { id, .. }
            | Self::FaceSwapAndEnhance { id, .. } => *id,
        }
    }

    pub fn kind(&self) -> StageKind {
        match self {
            Self::Enhance { .. } => StageKind::Enhance,
            Self::FaceBlur { .. } => StageKind::FaceBlur,
            Self::FaceSwapAndEnhance { .. } => StageKind::FaceSwapAndEnhance,
        }
    }

    pub fn as_enhance(&self) -> Option<&EnhanceConfig> {
        match self {
            Self::Enhance { config, .. } => Some(config),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageGraph {
    pub graph_schema_version: u32,
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

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(|err| {
            EngineError::InvariantViolation(format!(
                "Failed to read stage graph from {}: {err}",
                path.display()
            ))
        })?;
        Self::from_json_str(&data)
    }

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
        let mut swap_count = 0usize;

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
                StageConfig::FaceSwapAndEnhance { .. } => {
                    swap_count += 1;
                }
                StageConfig::FaceBlur { id, config } => {
                    if config.iterations == 0 {
                        return Err(EngineError::InvariantViolation(format!(
                            "StageGraph validation failed: stage {} {:?} requires iterations > 0",
                            id.0,
                            StageKind::FaceBlur
                        )));
                    }
                    if !(config.sigma.is_finite() && config.sigma > 0.0) {
                        return Err(EngineError::InvariantViolation(format!(
                            "StageGraph validation failed: stage {} {:?} requires sigma > 0",
                            id.0,
                            StageKind::FaceBlur
                        )));
                    }
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
        if swap_count > 1 {
            return Err(EngineError::InvariantViolation(
                "StageGraph validation failed: unsupported combination with multiple FaceSwapAndEnhance stages"
                    .into(),
            ));
        }

        Ok(())
    }

    pub fn single_enhance_config(&self) -> Option<&EnhanceConfig> {
        self.stages.iter().find_map(StageConfig::as_enhance)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditLevel {
    Pass,
    Warn,
    Fail,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditItem {
    pub level: AuditLevel,
    pub code: String,
    pub stage_id: Option<StageId>,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageTimingReport {
    pub decode_us: u64,
    pub preprocess_us: u64,
    pub infer_us: u64,
    pub postprocess_us: u64,
    pub encode_us: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineReport {
    pub selected_device: u32,
    pub provider: String,
    pub model_name: String,
    pub model_scale: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub frames_decoded: u64,
    pub frames_encoded: u64,
    pub stage_timing: StageTimingReport,
    pub stage_checksums: Vec<String>,
    pub vram_current_bytes: usize,
    pub vram_peak_bytes: usize,
    pub audit: Vec<AuditItem>,
}

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
            stages: vec![StageConfig::FaceBlur {
                id: StageId(1),
                config: BlurConfig::default(),
            }],
        };
        let err = graph
            .validate()
            .expect_err("must fail without enhance stage");
        assert!(err.to_string().contains("missing enhance stage"));
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
    fn graph_validation_rejects_unsupported_combination() {
        let graph = StageGraph {
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            stages: vec![
                enhance_stage(1),
                StageConfig::FaceSwapAndEnhance {
                    id: StageId(2),
                    config: SwapConfig::default(),
                },
                StageConfig::FaceSwapAndEnhance {
                    id: StageId(3),
                    config: SwapConfig::default(),
                },
            ],
        };
        let err = graph
            .validate()
            .expect_err("must fail on unsupported combination");
        assert!(err.to_string().contains("unsupported combination"));
    }

    #[test]
    fn graph_validation_accepts_single_enhance_chain() {
        let graph = StageGraph {
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            stages: vec![
                StageConfig::FaceBlur {
                    id: StageId(1),
                    config: BlurConfig::default(),
                },
                enhance_stage(2),
            ],
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
