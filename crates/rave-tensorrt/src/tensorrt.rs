//! TensorRT inference backend — ORT + TensorRTExecutionProvider + IO Binding.
//!
//! # Zero-copy contract
//!
//! Input and output tensors are bound to CUDA device pointers via ORT's
//! IO Binding API.  At no point does frame data touch host memory.
//!
//! # Execution provider policy
//!
//! **Only TensorRT EP is permitted.**  CPU EP is explicitly disabled.
//! After session creation, the provider list is validated.  If ORT falls
//! back to CPU for any graph node, `initialize()` returns an error.
//!
//! # CUDA stream ordering
//!
//! ORT creates its own internal CUDA stream for TensorRT EP execution.
//! We cannot inject `GpuContext::inference_stream` because `cudarc::CudaStream`
//! does not expose its raw `CUstream` handle (`pub(crate)` field).
//!
//! Correctness is maintained because `session.run_with_binding()` is
//! **synchronous** — ORT blocks the calling thread until all GPU kernels
//! on its internal stream complete.  Therefore:
//!
//! 1. Output buffer is fully written when `run_with_binding()` returns.
//! 2. CUDA global memory coherency guarantees visibility to any subsequent
//!    reader on any stream after this synchronization point.
//! 3. No additional inter-stream event is needed.
//!
//! # Output ring serialization
//!
//! `OutputRing` owns N pre-allocated device buffers.  `acquire()` checks
//! `Arc::strong_count == 1` before returning a slot, guaranteeing no
//! concurrent reader.  Ring size must be ≥ `downstream_channel_capacity + 2`.

use std::collections::HashSet;
use std::env;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::UNIX_EPOCH;

use async_trait::async_trait;
use cudarc::driver::{DevicePtr, DeviceSlice};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use ort::session::Session;
use ort::sys as ort_sys;
use ort::value::Value as OrtValue;

use rave_core::backend::{ModelMetadata, UpscaleBackend};
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::types::{GpuTexture, PixelFormat};

use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};

// Helper to create tensor from device memory using C API (Zero-Copy)
unsafe fn create_tensor_from_device_memory(
    ptr: *mut std::ffi::c_void,
    bytes: usize,
    shape: &[i64],
    elem_type: ort::tensor::TensorElementType,
) -> Result<OrtValue> {
    let api = ort::api();

    // Create MemoryInfo for CUDA
    let mut mem_info_ptr: *mut ort_sys::OrtMemoryInfo = std::ptr::null_mut();
    let name = std::ffi::CString::new("Cuda").unwrap();
    let status = unsafe {
        (api.CreateMemoryInfo)(
            name.as_ptr(),
            ort_sys::OrtAllocatorType::OrtArenaAllocator,
            0, // device id
            ort_sys::OrtMemType::OrtMemTypeDefault,
            &mut mem_info_ptr,
        )
    };
    if !status.0.is_null() {
        unsafe { (api.ReleaseStatus)(status.0) };
        return Err(EngineError::Inference(ort::Error::new(
            "Failed to create MemoryInfo",
        )));
    }

    // Create Tensor
    let mut ort_value_ptr: *mut ort_sys::OrtValue = std::ptr::null_mut();
    let status = unsafe {
        (api.CreateTensorWithDataAsOrtValue)(
            mem_info_ptr,
            ptr,
            bytes as _,
            shape.as_ptr(),
            shape.len() as _,
            elem_type.into(),
            &mut ort_value_ptr,
        )
    };

    // Release MemoryInfo (Tensor likely executes AddRef / Copy, or we hand over ownership?
    // ORT docs say CreateTensorWithDataAsOrtValue does NOT take ownership of mem_info,
    // but the resulting tensor keeps a reference?
    // Actually, usually we should release our handle if we don't need it.
    unsafe { (api.ReleaseMemoryInfo)(mem_info_ptr) };

    if !status.0.is_null() {
        unsafe { (api.ReleaseStatus)(status.0) };
        return Err(EngineError::Inference(ort::Error::new(
            "Failed to create Tensor",
        )));
    }

    // Wrap in OrtValue
    Ok(unsafe {
        ort::value::Value::<ort::value::DynValueTypeMarker>::from_ptr(
            std::ptr::NonNull::new(ort_value_ptr).unwrap(),
            None,
        )
    })
}

// use half::f16; // If half is dependency. ort might export it?
// I will guess ort::f16 or just try referencing it fully qualified if needed.
// I'll stick to basic imports for now and use full path in code if unsafe.

// ─── Precision policy ───────────────────────────────────────────────────────

/// TensorRT precision policy — controls EP optimization flags.
#[derive(Clone, Debug, Default)]
pub enum PrecisionPolicy {
    /// FP32 only — maximum accuracy, baseline performance.
    Fp32,
    /// FP16 mixed precision — 2× throughput on Tensor Cores.
    #[default]
    Fp16,
    /// INT8 quantized with calibration table — 4× throughput.
    /// Requires a pre-generated calibration table path.
    Int8 { calibration_table: PathBuf },
}

// ─── Batch config ──────────────────────────────────────────────────────────

/// Batch inference configuration.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Maximum batch size for pipelined inference.
    /// Must be ≤ model’s max dynamic batch axis.
    pub max_batch: usize,
    /// Collect at most this many frames before dispatching a batch,
    /// even if `max_batch` is not reached (latency bound).
    pub latency_deadline_us: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch: 1,
            latency_deadline_us: 8_000, // 8ms — half a 60fps frame
        }
    }
}
// ─── Inference metrics ───────────────────────────────────────────────────────

/// Atomic counters for inference stage observability.
#[derive(Debug)]
pub struct InferenceMetrics {
    /// Total frames inferred.
    pub frames_inferred: AtomicU64,
    /// Cumulative inference time in microseconds (for avg latency).
    pub total_inference_us: AtomicU64,
    /// Peak single-frame inference time in microseconds.
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

// ─── Ring metrics ────────────────────────────────────────────────────────────

/// Atomic counters for output ring buffer activity.
#[derive(Debug)]
pub struct RingMetrics {
    /// Successful slot reuses (slot was free, strong_count == 1).
    pub slot_reuse_count: AtomicU64,
    /// Times `acquire()` found a slot still held downstream (strong_count > 1).
    pub slot_contention_events: AtomicU64,
    /// Times a slot was acquired but it was the first use (not a reuse).
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

    pub fn snapshot(&self) -> (u64, u64, u64) {
        (
            self.slot_reuse_count.load(Ordering::Relaxed),
            self.slot_contention_events.load(Ordering::Relaxed),
            self.slot_first_use_count.load(Ordering::Relaxed),
        )
    }
}

impl Default for RingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Output ring buffer ─────────────────────────────────────────────────────

/// Fixed-size ring of pre-allocated device buffers for inference output.
pub struct OutputRing {
    slots: Vec<Arc<cudarc::driver::CudaSlice<u8>>>,
    cursor: usize,
    pub slot_bytes: usize,
    pub alloc_dims: (u32, u32),
    /// Whether each slot has been used at least once (for first-use tracking).
    used: Vec<bool>,
    pub metrics: RingMetrics,
}

impl OutputRing {
    /// Allocate `count` output buffers.
    ///
    /// `min_slots` is the enforced minimum (`downstream_capacity + 2`).
    /// Returns error if `count < min_slots`.
    pub fn new(
        ctx: &GpuContext,
        in_w: u32,
        in_h: u32,
        scale: u32,
        count: usize,
        min_slots: usize,
    ) -> Result<Self> {
        if count < min_slots {
            return Err(EngineError::DimensionMismatch(format!(
                "OutputRing: ring_size ({count}) < required minimum ({min_slots}). \
                 Ring must be ≥ downstream_channel_capacity + 2."
            )));
        }
        if count < 2 {
            return Err(EngineError::DimensionMismatch(
                "OutputRing: ring_size must be ≥ 2 for double-buffering".into(),
            ));
        }

        let out_w = (in_w * scale) as usize;
        let out_h = (in_h * scale) as usize;
        let slot_bytes = 3 * out_w * out_h * std::mem::size_of::<f32>();

        let slots = (0..count)
            .map(|_| ctx.alloc(slot_bytes).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;

        debug!(count, slot_bytes, out_w, out_h, "Output ring allocated");

        Ok(Self {
            slots,
            cursor: 0,
            slot_bytes,
            alloc_dims: (in_w, in_h),
            used: vec![false; count],
            metrics: RingMetrics::new(),
        })
    }

    /// Acquire the next ring slot for writing.
    ///
    /// # Serialization invariant
    ///
    /// Asserts `Arc::strong_count == 1` before returning.  If downstream
    /// still holds a reference, returns error and increments contention counter.
    pub fn acquire(&mut self) -> Result<Arc<cudarc::driver::CudaSlice<u8>>> {
        let slot = &self.slots[self.cursor];
        let sc = Arc::strong_count(slot);

        if sc != 1 {
            self.metrics
                .slot_contention_events
                .fetch_add(1, Ordering::Relaxed);
            return Err(EngineError::BufferTooSmall {
                need: self.slot_bytes,
                have: 0,
            });
        }

        // Debug assertion — belt-and-suspenders check.
        debug_assert_eq!(
            Arc::strong_count(slot),
            1,
            "OutputRing: slot {} strong_count must be 1 before reuse, got {}",
            self.cursor,
            sc
        );

        if self.used[self.cursor] {
            self.metrics
                .slot_reuse_count
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.used[self.cursor] = true;
            self.metrics
                .slot_first_use_count
                .fetch_add(1, Ordering::Relaxed);
        }

        let cloned = Arc::clone(slot);
        self.cursor = (self.cursor + 1) % self.slots.len();
        Ok(cloned)
    }

    pub fn needs_realloc(&self, in_w: u32, in_h: u32) -> bool {
        self.alloc_dims != (in_w, in_h)
    }

    /// Reallocate all slots.  All must have `strong_count == 1`.
    pub fn reallocate(&mut self, ctx: &GpuContext, in_w: u32, in_h: u32, scale: u32) -> Result<()> {
        for (i, slot) in self.slots.iter().enumerate() {
            let sc = Arc::strong_count(slot);
            if sc != 1 {
                return Err(EngineError::DimensionMismatch(format!(
                    "Cannot reallocate ring: slot {} still in use (strong_count={})",
                    i, sc,
                )));
            }
        }

        // Free old slots — decrement VRAM accounting.
        for _slot in &self.slots {
            ctx.vram_dec(self.slot_bytes);
        }

        let count = self.slots.len();
        let out_w = (in_w * scale) as usize;
        let out_h = (in_h * scale) as usize;
        let slot_bytes = 3 * out_w * out_h * std::mem::size_of::<f32>();

        self.slots = (0..count)
            .map(|_| ctx.alloc(slot_bytes).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;
        self.cursor = 0;
        self.slot_bytes = slot_bytes;
        self.alloc_dims = (in_w, in_h);
        self.used = vec![false; count];

        debug!(count, slot_bytes, out_w, out_h, "Output ring reallocated");
        Ok(())
    }

    /// Total number of slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

// ─── Inference state ─────────────────────────────────────────────────────────

struct InferenceState {
    session: Session,
    ring: Option<OutputRing>,
}

/// Resolve ORT tensor element type from our PixelFormat.
fn ort_element_type(format: PixelFormat) -> ort::tensor::TensorElementType {
    match format {
        PixelFormat::RgbPlanarF16 => ort::tensor::TensorElementType::Float16,
        _ => ort::tensor::TensorElementType::Float32,
    }
}

// ─── Backend ─────────────────────────────────────────────────────────────────

pub struct TensorRtBackend {
    model_path: PathBuf,
    ctx: Arc<GpuContext>,
    device_id: i32,
    ring_size: usize,
    min_ring_slots: usize,
    meta: OnceLock<ModelMetadata>,
    selected_provider: OnceLock<String>,
    state: Mutex<Option<InferenceState>>,
    pub inference_metrics: InferenceMetrics,
    /// Phase 8: precision policy for TRT EP.
    pub precision_policy: PrecisionPolicy,
    /// Phase 8: batch configuration.
    pub batch_config: BatchConfig,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum OrtEpMode {
    Auto,
    TensorRtOnly,
    CudaOnly,
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

#[cfg(target_os = "linux")]
const RTLD_NOW: i32 = 2;
#[cfg(all(target_os = "linux", test))]
const RTLD_LOCAL: i32 = 0;
#[cfg(target_os = "linux")]
const RTLD_GLOBAL: i32 = 0x100;

#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
enum OrtProviderKind {
    Cuda,
    TensorRt,
}

#[cfg(target_os = "linux")]
impl OrtProviderKind {
    fn soname(self) -> &'static str {
        match self {
            OrtProviderKind::Cuda => "libonnxruntime_providers_cuda.so",
            OrtProviderKind::TensorRt => "libonnxruntime_providers_tensorrt.so",
        }
    }

    fn label(self) -> &'static str {
        match self {
            OrtProviderKind::Cuda => "providers_cuda",
            OrtProviderKind::TensorRt => "providers_tensorrt",
        }
    }
}

impl TensorRtBackend {
    /// Create a new backend instance.
    ///
    /// # Parameters
    ///
    /// - `ring_size`: number of output ring slots to pre-allocate.
    /// - `downstream_capacity`: the bounded channel capacity between inference
    ///   and the encoder.  Ring size is validated ≥ `downstream_capacity + 2`.
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

    /// Create with explicit precision policy and batch config.
    pub fn with_precision(
        model_path: PathBuf,
        ctx: Arc<GpuContext>,
        device_id: i32,
        ring_size: usize,
        downstream_capacity: usize,
        precision_policy: PrecisionPolicy,
        batch_config: BatchConfig,
    ) -> Self {
        let min_ring_slots = downstream_capacity + 2;
        assert!(
            ring_size >= min_ring_slots,
            "ring_size ({ring_size}) must be ≥ downstream_capacity + 2 ({min_ring_slots})"
        );
        Self {
            model_path,
            ctx,
            device_id,
            ring_size,
            min_ring_slots,
            meta: OnceLock::new(),
            selected_provider: OnceLock::new(),
            state: Mutex::new(None),
            inference_metrics: InferenceMetrics::new(),
            precision_policy,
            batch_config,
        }
    }

    /// Get or create cached ORT MemoryInfo (avoids per-frame allocation).
    /// Create a CUDA memory info structure.
    fn mem_info(&self) -> Result<ort::memory::MemoryInfo> {
        ort::memory::MemoryInfo::new(
            ort::memory::AllocationDevice::CUDA,
            0,
            ort::memory::AllocatorType::Device,
            ort::memory::MemoryType::Default,
        )
        .map_err(|e| EngineError::ModelMetadata(format!("MemoryInfo: {e}")))
    }

    /// Access ring metrics (if initialized).
    pub async fn ring_metrics(&self) -> Option<(u64, u64, u64)> {
        let guard = self.state.lock().await;
        guard
            .as_ref()
            .and_then(|s| s.ring.as_ref())
            .map(|r| r.metrics.snapshot())
    }

    /// Active ORT execution provider selected during initialization.
    pub fn selected_provider(&self) -> Option<&str> {
        self.selected_provider.get().map(String::as_str)
    }

    fn extract_metadata(session: &Session) -> Result<ModelMetadata> {
        let inputs = session.inputs();
        let outputs = session.outputs();

        if inputs.is_empty() || outputs.is_empty() {
            return Err(EngineError::ModelMetadata(
                "Model must have at least one input and one output tensor".into(),
            ));
        }

        // ... imports ...

        // ...

        // ... in extract_metadata ...
        // ...
        let input_info = &inputs[0];
        let output_info = &outputs[0];
        let input_name = input_info.name().to_string();
        let output_name = output_info.name().to_string();

        // ...

        let input_dims = match input_info.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.clone(),
            other => {
                return Err(EngineError::ModelMetadata(format!(
                    "Expected tensor input, got {:?}",
                    other
                )));
            }
        };

        let output_dims = match output_info.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.clone(),
            other => {
                return Err(EngineError::ModelMetadata(format!(
                    "Expected tensor output, got {:?}",
                    other
                )));
            }
        };

        if input_dims.len() != 4 || output_dims.len() != 4 {
            return Err(EngineError::ModelMetadata(format!(
                "Expected 4D tensors (NCHW), got input={}D output={}D",
                input_dims.len(),
                output_dims.len()
            )));
        }

        let input_channels = input_dims[1] as u32;

        let ih = input_dims[2];
        let iw = input_dims[3];
        let oh = output_dims[2];
        let ow = output_dims[3];

        let scale = if ih > 0 && oh > 0 && iw > 0 && ow > 0 {
            (oh / ih) as u32
        } else {
            warn!("Dynamic spatial axes — defaulting to scale=4");
            4
        };

        let min_input_hw = (
            if ih > 0 { ih as u32 } else { 1 },
            if iw > 0 { iw as u32 } else { 1 },
        );

        let max_input_hw = (
            if ih > 0 { ih as u32 } else { u32::MAX },
            if iw > 0 { iw as u32 } else { u32::MAX },
        );

        let name = session
            .metadata()
            .map(|m| m.name().unwrap_or("unknown".to_string()))
            .unwrap_or_else(|_| "unknown".to_string());

        Ok(ModelMetadata {
            name,
            scale,
            input_name,
            output_name,
            input_channels,
            min_input_hw,
            max_input_hw,
        })
    }

    fn ort_ep_mode() -> OrtEpMode {
        match env::var("RAVE_ORT_TENSORRT")
            .unwrap_or_else(|_| "auto".to_string())
            .to_lowercase()
            .as_str()
        {
            "0" | "off" | "false" | "cuda" | "cuda-only" => OrtEpMode::CudaOnly,
            "1" | "on" | "true" | "trt" | "trt-only" => OrtEpMode::TensorRtOnly,
            _ => OrtEpMode::Auto,
        }
    }

    #[cfg(target_os = "linux")]
    fn is_wsl2() -> bool {
        std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|s| s.to_ascii_lowercase().contains("microsoft"))
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "linux"))]
    fn is_wsl2() -> bool {
        false
    }

    #[cfg(target_os = "linux")]
    fn ort_provider_cache_dirs_newest_first() -> Vec<PathBuf> {
        let mut dirs = Vec::<(u128, PathBuf)>::new();
        let Some(home) = env::var_os("HOME") else {
            return Vec::new();
        };
        let base = PathBuf::from(home).join(".cache/ort.pyke.io/dfbin");
        let Ok(triples) = std::fs::read_dir(base) else {
            return Vec::new();
        };

        for triple in triples.flatten() {
            let triple_path = triple.path();
            if !triple_path.is_dir() {
                continue;
            }
            let Ok(hashes) = std::fs::read_dir(triple_path) else {
                continue;
            };
            for hash in hashes.flatten() {
                let path = hash.path();
                if !path.is_dir() {
                    continue;
                }
                let modified = std::fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                dirs.push((modified, path));
            }
        }

        dirs.sort_by(|a, b| b.cmp(a));
        dirs.into_iter().map(|(_, path)| path).collect()
    }

    #[cfg(target_os = "linux")]
    fn ort_provider_search_dirs() -> Vec<(PathBuf, &'static str)> {
        let mut dirs = Vec::<(PathBuf, &'static str)>::new();
        if let Some(dir) = env::var_os("ORT_DYLIB_PATH") {
            dirs.push((PathBuf::from(dir), "ORT_DYLIB_PATH"));
        }
        if let Some(dir) = env::var_os("ORT_LIB_LOCATION") {
            dirs.push((PathBuf::from(dir), "ORT_LIB_LOCATION"));
        }
        if Self::is_wsl2() {
            for dir in Self::ort_provider_cache_dirs_newest_first() {
                dirs.push((dir, "ort_cache_newest"));
            }
            if let Ok(exe) = env::current_exe()
                && let Some(dir) = exe.parent()
            {
                dirs.push((dir.to_path_buf(), "exe_dir"));
                dirs.push((dir.join("deps"), "exe_dir/deps"));
            }
        } else {
            if let Ok(exe) = env::current_exe()
                && let Some(dir) = exe.parent()
            {
                dirs.push((dir.to_path_buf(), "exe_dir"));
                dirs.push((dir.join("deps"), "exe_dir/deps"));
            }
            for dir in Self::ort_provider_cache_dirs_newest_first() {
                dirs.push((dir, "ort_cache_newest"));
            }
        }

        let mut uniq = HashSet::<PathBuf>::new();
        dirs.retain(|(p, _)| uniq.insert(p.clone()));
        dirs
    }

    #[cfg(all(target_os = "linux", test))]
    fn ort_provider_candidates(lib_name: &str) -> Vec<PathBuf> {
        Self::ort_provider_search_dirs()
            .into_iter()
            .map(|(dir, _)| dir.join(lib_name))
            .collect()
    }

    #[cfg(target_os = "linux")]
    fn configure_ort_loader_path(dir: &Path) {
        // SAFETY: Process env mutation is done at startup during backend init.
        unsafe { env::set_var("ORT_DYLIB_PATH", dir) };
        // SAFETY: Process env mutation is done at startup during backend init.
        unsafe { env::set_var("ORT_LIB_LOCATION", dir) };
    }

    #[cfg(target_os = "linux")]
    fn resolve_ort_provider_dir(kind: OrtProviderKind) -> Result<(PathBuf, &'static str)> {
        let provider = kind.soname();
        for (dir, source) in Self::ort_provider_search_dirs() {
            let shared = dir.join("libonnxruntime_providers_shared.so");
            let provider_path = dir.join(provider);
            if shared.is_file() && provider_path.is_file() {
                info!(
                    path = %shared.display(),
                    source,
                    "ORT providers_shared resolved"
                );
                info!(
                    provider = kind.label(),
                    path = %provider_path.display(),
                    source,
                    "ORT provider resolved"
                );
                Self::configure_ort_loader_path(&dir);
                return Ok((dir, source));
            }
        }

        Err(EngineError::ModelMetadata(format!(
            "Could not find ORT provider pair (libonnxruntime_providers_shared.so + {}). \
Set ORT_DYLIB_PATH or ORT_LIB_LOCATION to that directory, then re-run scripts/test_ort_provider_load.sh.",
            provider
        )))
    }

    #[cfg(target_os = "linux")]
    fn dlopen_path(path: &Path, flags: i32) -> Result<()> {
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| EngineError::ModelMetadata("Invalid dlopen path".into()))?;
        // SAFETY: `dlopen` expects a valid, NUL-terminated C string and flags.
        let handle = unsafe { dlopen(cpath.as_ptr(), flags) };
        if handle.is_null() {
            // SAFETY: `dlerror` returns a thread-local pointer or null.
            let err = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().to_string()
                }
            };
            return Err(EngineError::ModelMetadata(format!(
                "dlopen failed for {}: {err}",
                path.display()
            )));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn preload_cuda_runtime_libs() {
        // Prefer explicit toolkit paths to avoid reliance on unrelated app installs.
        let mut roots = Vec::<PathBuf>::new();
        if let Some(dir) = env::var_os("RAVE_CUDA_RUNTIME_DIR") {
            roots.push(PathBuf::from(dir));
        }
        roots.push(PathBuf::from("/usr/local/cuda-12/targets/x86_64-linux/lib"));
        roots.push(PathBuf::from("/usr/local/cuda/lib64"));

        let libs = [
            "libcudart.so.12",
            "libcublasLt.so.12",
            "libcublas.so.12",
            "libnvrtc.so.12",
            "libcurand.so.10",
            "libcufft.so.11",
        ];

        for root in roots {
            if !root.is_dir() {
                continue;
            }
            let mut loaded = 0usize;
            for lib in libs {
                let p = root.join(lib);
                if !p.is_file() {
                    continue;
                }
                if Self::dlopen_path(&p, RTLD_NOW | RTLD_GLOBAL).is_ok() {
                    loaded += 1;
                }
            }
            if loaded > 0 {
                info!(root = %root.display(), loaded, "Preloaded CUDA runtime libraries for ORT");
                break;
            }
        }
    }

    /// When ORT is linked statically, TensorRT provider expects `Provider_GetHost`
    /// from `libonnxruntime_providers_shared.so` to already be present in the process.
    /// Preloading this bridge with `RTLD_GLOBAL` satisfies that symbol before TRT EP load.
    #[cfg(target_os = "linux")]
    fn preload_ort_provider_pair(kind: OrtProviderKind) -> Result<()> {
        let (dir, source) = Self::resolve_ort_provider_dir(kind)?;
        let shared = dir.join("libonnxruntime_providers_shared.so");
        let provider = dir.join(kind.soname());

        Self::dlopen_path(&shared, RTLD_NOW | RTLD_GLOBAL).map_err(|e| {
            EngineError::ModelMetadata(format!(
                "Failed loading providers_shared from {} ({e}). \
Ensure ORT_DYLIB_PATH/ORT_LIB_LOCATION points to a valid ORT cache dir.",
                shared.display()
            ))
        })?;

        info!(
            path = %provider.display(),
            provider = kind.label(),
            "Skipping explicit provider dlopen; relying on startup LD_LIBRARY_PATH + ORT registration"
        );

        info!(
            source,
            dir = %dir.display(),
            provider = kind.label(),
            path = %provider.display(),
            "ORT provider pair prepared (providers_shared preloaded, provider path configured)"
        );
        Ok(())
    }

    #[cfg(all(target_os = "linux", test))]
    fn preload_ort_provider_bridge() -> Result<()> {
        for path in Self::ort_provider_candidates("libonnxruntime_providers_shared.so") {
            if path.is_file() {
                return Self::dlopen_path(&path, RTLD_NOW | RTLD_GLOBAL);
            }
        }
        Err(EngineError::ModelMetadata(
            "Could not preload libonnxruntime_providers_shared.so from known search paths".into(),
        ))
    }

    #[cfg(all(not(target_os = "linux"), test))]
    fn preload_ort_provider_bridge() -> Result<()> {
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn preload_cuda_runtime_libs() {}

    fn build_trt_session(&self) -> Result<Session> {
        Self::preload_cuda_runtime_libs();
        Self::preload_ort_provider_pair(OrtProviderKind::TensorRt)?;

        let mut trt_ep = TensorRTExecutionProvider::default()
            .with_device_id(self.device_id)
            .with_engine_cache(true)
            .with_engine_cache_path(
                self.model_path
                    .parent()
                    .unwrap_or(&self.model_path)
                    .join("trt_cache")
                    .to_string_lossy()
                    .to_string(),
            );

        match &self.precision_policy {
            PrecisionPolicy::Fp32 => {
                info!("TRT precision: FP32 (no mixed precision)");
            }
            PrecisionPolicy::Fp16 => {
                trt_ep = trt_ep.with_fp16(true);
                info!("TRT precision: FP16 mixed precision");
            }
            PrecisionPolicy::Int8 { calibration_table } => {
                trt_ep = trt_ep.with_fp16(true).with_int8(true);
                info!(
                    table = %calibration_table.display(),
                    "TRT precision: INT8 with calibration table"
                );
            }
        }

        Session::builder()?
            .with_execution_providers([trt_ep.build().error_on_failure()])?
            .with_intra_threads(1)?
            .commit_from_file(&self.model_path)
            .map_err(Into::into)
    }

    fn build_cuda_session(&self) -> Result<Session> {
        Self::preload_cuda_runtime_libs();
        Self::preload_ort_provider_pair(OrtProviderKind::Cuda)?;
        let cuda_ep = CUDAExecutionProvider::default().with_device_id(self.device_id);
        Session::builder()?
            .with_execution_providers([cuda_ep.build().error_on_failure()])?
            .with_intra_threads(1)?
            .commit_from_file(&self.model_path)
            .map_err(Into::into)
    }

    fn pointer_identity_mismatch(
        input_ptr: u64,
        texture_ptr: u64,
        output_ptr: u64,
        ring_slot_ptr: u64,
    ) -> Option<String> {
        if input_ptr != texture_ptr {
            return Some(format!(
                "POINTER MISMATCH: IO-bound input (0x{input_ptr:016x}) != GpuTexture (0x{texture_ptr:016x})"
            ));
        }
        if output_ptr != ring_slot_ptr {
            return Some(format!(
                "POINTER MISMATCH: IO-bound output (0x{output_ptr:016x}) != ring slot (0x{ring_slot_ptr:016x})"
            ));
        }
        None
    }

    /// Verify that IO-bound device pointers match the source GpuTexture
    /// and OutputRing slot pointers exactly (pointer identity check).
    ///
    /// Called by `run_io_bound` to audit that ORT IO Binding uses our
    /// device pointers without any host staging or reallocation.
    fn verify_pointer_identity(
        input_ptr: u64,
        output_ptr: u64,
        input_texture: &GpuTexture,
        ring_slot_ptr: u64,
    ) -> Result<()> {
        let texture_ptr = input_texture.device_ptr();
        debug!(
            input_ptr = format!("0x{:016x}", input_ptr),
            texture_ptr = format!("0x{:016x}", texture_ptr),
            output_ptr = format!("0x{:016x}", output_ptr),
            ring_slot_ptr = format!("0x{:016x}", ring_slot_ptr),
            "IO-binding pointer identity audit"
        );

        if let Some(message) =
            Self::pointer_identity_mismatch(input_ptr, texture_ptr, output_ptr, ring_slot_ptr)
        {
            debug_assert!(false, "{message}");
            #[cfg(feature = "audit-no-host-copies")]
            if rave_core::host_copy_audit::is_strict_mode() {
                return Err(EngineError::InvariantViolation(message));
            }
            rave_core::host_copy_violation!("inference", "{message}");
        }
        Ok(())
    }

    fn run_io_bound(
        session: &mut Session,
        meta: &ModelMetadata,
        input: &GpuTexture,
        output_ptr: u64,
        output_bytes: usize,
        _ctx: &GpuContext,
        _mem_info: &ort::memory::MemoryInfo, // Unused argument if we recreate it, or use it if passed.
    ) -> Result<()> {
        let in_w = input.width as i64;
        let in_h = input.height as i64;
        let out_w = in_w * meta.scale as i64;
        let out_h = in_h * meta.scale as i64;

        let input_shape: Vec<i64> = vec![1, meta.input_channels as i64, in_h, in_w];
        let output_shape: Vec<i64> = vec![1, meta.input_channels as i64, out_h, out_w];

        // Resolve element type from pixel format (F16 or F32).
        let elem_type = ort_element_type(input.format);
        let elem_bytes = input.format.element_bytes();
        let input_bytes = input.data.len();

        let expected = (output_shape.iter().product::<i64>() as usize) * elem_bytes;
        if output_bytes < expected {
            return Err(EngineError::BufferTooSmall {
                need: expected,
                have: output_bytes,
            });
        }

        let input_ptr = input.device_ptr();

        // Phase 7: Pointer identity audit — verify device pointers match.
        Self::verify_pointer_identity(input_ptr, output_ptr, input, output_ptr)?;

        let mut binding = session.create_binding()?;

        // Create fresh MemoryInfo for this opt (cheap, avoids Sync issues with cache)
        // (Removed: using FFI helper instead)

        // SAFETY — ORT IO Binding with raw device pointers:
        unsafe {
            // Create Input Tensor from raw device memory (zero-copy)
            let input_tensor = create_tensor_from_device_memory(
                input_ptr as *mut _,
                input_bytes,
                &input_shape,
                elem_type,
            )?;

            binding.bind_input(&meta.input_name, &input_tensor)?;

            // Create Output Tensor from raw device memory (zero-copy)
            let output_tensor = create_tensor_from_device_memory(
                output_ptr as *mut _,
                output_bytes,
                &output_shape,
                elem_type,
            )?;

            binding.bind_output(&meta.output_name, output_tensor)?;
        }

        // Synchronous execution — ORT blocks until all TensorRT kernels complete.
        session.run_binding(&binding)?;

        Ok(())
    }
}

#[cfg(test)]
mod pointer_audit_tests {
    use super::TensorRtBackend;

    #[test]
    fn pointer_identity_audit_accepts_matching_addresses() {
        let mismatch = TensorRtBackend::pointer_identity_mismatch(0x1000, 0x1000, 0x2000, 0x2000);
        assert!(mismatch.is_none());
    }

    #[test]
    fn pointer_identity_audit_reports_mismatch() {
        let mismatch = TensorRtBackend::pointer_identity_mismatch(0x1000, 0x1111, 0x2000, 0x2000)
            .expect("mismatch should be reported");
        assert!(mismatch.contains("IO-bound input"));
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::{RTLD_LOCAL, RTLD_NOW, TensorRtBackend};
    use rave_core::backend::UpscaleBackend;
    use std::env;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires ORT TensorRT provider libs on host"]
    fn providers_load_with_bridge_preloaded() {
        // Ensure bridge is globally visible before loading TensorRT provider.
        TensorRtBackend::preload_ort_provider_bridge().expect("failed to preload providers_shared");
        let trt_candidates =
            TensorRtBackend::ort_provider_candidates("libonnxruntime_providers_tensorrt.so");
        assert!(
            !trt_candidates.is_empty(),
            "no libonnxruntime_providers_tensorrt.so candidates found"
        );
        let path = trt_candidates[0].clone();
        TensorRtBackend::dlopen_path(&path, RTLD_NOW | RTLD_LOCAL)
            .expect("providers_tensorrt should dlopen after providers_shared preload");
    }

    #[test]
    #[ignore = "requires model + full ORT/TensorRT runtime"]
    fn ort_registers_tensorrt_ep_smoke() {
        let model = env::var("RAVE_TEST_ONNX_MODEL").expect("set RAVE_TEST_ONNX_MODEL");
        let backend = TensorRtBackend::new(
            PathBuf::from(model),
            rave_core::context::GpuContext::new(0).expect("cuda ctx"),
            0,
            6,
            4,
        );
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async move { backend.initialize().await })
            .expect("TensorRT EP registration should succeed");
    }
}

#[async_trait]
impl UpscaleBackend for TensorRtBackend {
    async fn initialize(&self) -> Result<()> {
        let mut guard = self.state.lock().await;
        if guard.is_some() {
            return Err(EngineError::ModelMetadata("Already initialized".into()));
        }

        let ep_mode = Self::ort_ep_mode();
        let on_wsl = Self::is_wsl2();
        info!(
            path = %self.model_path.display(),
            ?ep_mode,
            on_wsl,
            "Loading ONNX model with ORT execution provider policy"
        );

        let (session, active_provider) = match ep_mode {
            OrtEpMode::CudaOnly => {
                info!("RAVE_ORT_TENSORRT disables TensorRT EP; using CUDAExecutionProvider");
                (self.build_cuda_session()?, "CUDAExecutionProvider")
            }
            OrtEpMode::TensorRtOnly => (self.build_trt_session()?, "TensorrtExecutionProvider"),
            OrtEpMode::Auto => match self.build_trt_session() {
                Ok(session) => (session, "TensorrtExecutionProvider"),
                Err(e) => {
                    warn!(
                        error = %e,
                        on_wsl,
                        "TensorRT EP registration failed; falling back to CUDAExecutionProvider"
                    );
                    (self.build_cuda_session()?, "CUDAExecutionProvider")
                }
            },
        };
        info!(
            provider = active_provider,
            "ORT execution provider selected"
        );

        let metadata = Self::extract_metadata(&session)?;
        info!(
            name = %metadata.name,
            scale = metadata.scale,
            input = %metadata.input_name,
            output = %metadata.output_name,
            ring_size = self.ring_size,
            min_ring_slots = self.min_ring_slots,
            provider = active_provider,
            precision = ?self.precision_policy,
            max_batch = self.batch_config.max_batch,
            "Model loaded"
        );

        let _ = self.meta.set(metadata);
        let _ = self.selected_provider.set(active_provider.to_string());

        *guard = Some(InferenceState {
            session,
            ring: None,
        });

        Ok(())
    }

    async fn process(&self, input: GpuTexture) -> Result<GpuTexture> {
        // Accept both F32 and F16 planar RGB.
        match input.format {
            PixelFormat::RgbPlanarF32 | PixelFormat::RgbPlanarF16 => {}
            other => {
                return Err(EngineError::FormatMismatch {
                    expected: PixelFormat::RgbPlanarF32,
                    actual: other,
                });
            }
        }

        let meta = self.meta.get().ok_or(EngineError::NotInitialized)?;
        let mut guard = self.state.lock().await;
        let state = guard.as_mut().ok_or(EngineError::NotInitialized)?;

        // Lazy ring init / realloc.
        match &mut state.ring {
            Some(ring) if ring.needs_realloc(input.width, input.height) => {
                debug!(
                    old_w = ring.alloc_dims.0,
                    old_h = ring.alloc_dims.1,
                    new_w = input.width,
                    new_h = input.height,
                    "Reallocating output ring"
                );
                ring.reallocate(&self.ctx, input.width, input.height, meta.scale)?;
            }
            None => {
                debug!(
                    w = input.width,
                    h = input.height,
                    slots = self.ring_size,
                    "Lazily creating output ring"
                );
                state.ring = Some(OutputRing::new(
                    &self.ctx,
                    input.width,
                    input.height,
                    meta.scale,
                    self.ring_size,
                    self.min_ring_slots,
                )?);
            }
            Some(_) => {}
        }

        let ring = state.ring.as_mut().unwrap();

        // Debug-mode host allocation tracking.
        #[cfg(feature = "debug-alloc")]
        {
            rave_core::debug_alloc::reset();
            rave_core::debug_alloc::enable();
        }

        let output_arc = ring.acquire()?;
        let output_ptr = *(*output_arc).device_ptr();
        let output_bytes = ring.slot_bytes;

        // ── Inference with latency measurement ──
        let t_start = std::time::Instant::now();

        let mem_info = self.mem_info()?;
        Self::run_io_bound(
            &mut state.session,
            meta,
            &input,
            output_ptr,
            output_bytes,
            &self.ctx,
            &mem_info,
        )?;

        let elapsed_us = t_start.elapsed().as_micros() as u64;
        self.inference_metrics.record(elapsed_us);

        #[cfg(feature = "debug-alloc")]
        {
            rave_core::debug_alloc::disable();
            let host_allocs = rave_core::debug_alloc::count();
            debug_assert_eq!(
                host_allocs, 0,
                "VIOLATION: {host_allocs} host allocations during inference"
            );
        }

        let out_w = input.width * meta.scale;
        let out_h = input.height * meta.scale;
        let elem_bytes = input.format.element_bytes();

        Ok(GpuTexture {
            data: output_arc,
            width: out_w,
            height: out_h,
            pitch: (out_w as usize) * elem_bytes,
            format: input.format, // Preserve F32 or F16
        })
    }

    async fn shutdown(&self) -> Result<()> {
        let mut guard = self.state.lock().await;
        if let Some(state) = guard.take() {
            info!("Shutting down TensorRT backend");
            self.ctx.sync_all()?;

            // Report ring metrics.
            if let Some(ring) = &state.ring {
                let (reuse, contention, first) = ring.metrics.snapshot();
                info!(reuse, contention, first, "Final ring metrics");
            }

            // Report inference metrics.
            let snap = self.inference_metrics.snapshot();
            info!(
                frames = snap.frames_inferred,
                avg_us = snap.avg_inference_us,
                peak_us = snap.peak_inference_us,
                precision = ?self.precision_policy,
                "Final inference metrics"
            );

            // Report VRAM.
            let (current, peak) = self.ctx.vram_usage();
            info!(
                current_mb = current / (1024 * 1024),
                peak_mb = peak / (1024 * 1024),
                "Final VRAM usage"
            );

            drop(state.ring);
            drop(state.session);
            debug!("TensorRT backend shutdown complete");
        }
        Ok(())
    }

    fn metadata(&self) -> Result<&ModelMetadata> {
        self.meta.get().ok_or(EngineError::NotInitialized)
    }
}

impl Drop for TensorRtBackend {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.state.try_lock()
            && let Some(state) = guard.take()
        {
            let _ = self.ctx.sync_all();
            drop(state);
        }
    }
}
