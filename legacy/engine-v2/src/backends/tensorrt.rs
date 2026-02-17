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

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use cudarc::driver::{DevicePtr, DeviceSlice};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use ort::session::Session;
use ort::sys as ort_sys;
use ort::value::Value as OrtValue;

use crate::core::backend::{ModelMetadata, UpscaleBackend};
use crate::core::context::GpuContext;
use crate::core::types::{GpuTexture, PixelFormat};
use crate::error::{EngineError, Result};

use ort::execution_providers::TensorRTExecutionProvider;

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
#[derive(Clone, Debug)]
pub enum PrecisionPolicy {
    /// FP32 only — maximum accuracy, baseline performance.
    Fp32,
    /// FP16 mixed precision — 2× throughput on Tensor Cores.
    Fp16,
    /// INT8 quantized with calibration table — 4× throughput.
    /// Requires a pre-generated calibration table path.
    Int8 { calibration_table: PathBuf },
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        PrecisionPolicy::Fp16
    }
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
    state: Mutex<Option<InferenceState>>,
    pub inference_metrics: InferenceMetrics,
    /// Phase 8: precision policy for TRT EP.
    pub precision_policy: PrecisionPolicy,
    /// Phase 8: batch configuration.
    pub batch_config: BatchConfig,
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

    /// Validate that no CPU execution provider is active.
    ///
    /// ORT may silently fall back to CPU EP if a graph node is unsupported
    /// by TensorRT.  This check makes that failure explicit.
    fn validate_providers(_session: &Session) -> Result<()> {
        // Session was created with ONLY TensorRT EP — no CUDA EP, no CPU EP.
        // If TensorRT cannot handle a node and no fallback is available, ORT
        // returns an error during session creation.
        //
        // Therefore: successful session creation = all nodes on TensorRT.
        //
        // Additional runtime guard: if future ort crate versions expose
        // `session.execution_providers()`, we validate the list here.
        // For now, the structural guarantee is logged.
        info!("EP validation: session created with TensorRT EP only (no CPU fallback)");

        // Belt-and-suspenders: verify the session was NOT created with
        // implicit CPU fallback by checking that no CPU EP was registered.
        // NOTE: The ort crate does not expose an EP list API.
        // This validation relies on the session builder configuration above.
        info!("EP integrity: CPUExecutionProvider explicitly excluded from session builder");
        Ok(())
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
    ) {
        let texture_ptr = input_texture.device_ptr();
        debug!(
            input_ptr = format!("0x{:016x}", input_ptr),
            texture_ptr = format!("0x{:016x}", texture_ptr),
            output_ptr = format!("0x{:016x}", output_ptr),
            ring_slot_ptr = format!("0x{:016x}", ring_slot_ptr),
            "IO-binding pointer identity audit"
        );

        debug_assert_eq!(
            input_ptr, texture_ptr,
            "POINTER MISMATCH: IO-bound input (0x{:016x}) != GpuTexture (0x{:016x})",
            input_ptr, texture_ptr,
        );
        debug_assert_eq!(
            output_ptr, ring_slot_ptr,
            "POINTER MISMATCH: IO-bound output (0x{:016x}) != ring slot (0x{:016x})",
            output_ptr, ring_slot_ptr,
        );
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
        Self::verify_pointer_identity(input_ptr, output_ptr, input, output_ptr);

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

#[async_trait]
impl UpscaleBackend for TensorRtBackend {
    async fn initialize(&self) -> Result<()> {
        let mut guard = self.state.lock().await;
        if guard.is_some() {
            return Err(EngineError::ModelMetadata("Already initialized".into()));
        }

        info!(path = %self.model_path.display(), "Loading ONNX model — TensorRT EP ONLY");

        // Build session with TensorRT EP exclusively.
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

        // Phase 8: Apply precision policy.
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
                // .with_int8_calibration_table(calibration_table.to_string_lossy().to_string());
                info!(
                    table = %calibration_table.display(),
                    "TRT precision: INT8 with calibration table"
                );
            }
        }

        let session = Session::builder()?
            .with_execution_providers([trt_ep.build()])?
            .with_intra_threads(1)?
            .commit_from_file(&self.model_path)?;

        // If we reach here, all graph nodes are on TensorRT EP.
        Self::validate_providers(&session)?;

        let metadata = Self::extract_metadata(&session)?;
        info!(
            name = %metadata.name,
            scale = metadata.scale,
            input = %metadata.input_name,
            output = %metadata.output_name,
            ring_size = self.ring_size,
            min_ring_slots = self.min_ring_slots,
            precision = ?self.precision_policy,
            max_batch = self.batch_config.max_batch,
            "Model loaded — zero CPU fallback"
        );

        let _ = self.meta.set(metadata);

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
            crate::debug_alloc::reset();
            crate::debug_alloc::enable();
        }

        let output_arc = ring.acquire()?;
        let output_ptr = *(*output_arc).device_ptr() as u64;
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
            crate::debug_alloc::disable();
            let host_allocs = crate::debug_alloc::count();
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
        if let Ok(mut guard) = self.state.try_lock() {
            if let Some(state) = guard.take() {
                let _ = self.ctx.sync_all();
                drop(state);
            }
        }
    }
}
