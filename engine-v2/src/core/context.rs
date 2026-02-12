//! Shared CUDA context — single device, explicit stream management,
//! bucketed buffer pool, VRAM accounting, stream overlap timing,
//! hardware-aligned allocation, GPU profiling hooks, and
//! production metrics enforcement (Phase 9).
//!
//! # Buffer pool (Phase 6 — zero-free steady state)
//!
//! All device allocations go through [`GpuContext::alloc`] and recycled
//! buffers are returned via [`GpuContext::recycle`].  After warm-up, the
//! pool holds enough buffers to satisfy every frame without hitting the
//! CUDA driver allocator.
//!
//! ## Bucketing strategy
//!
//! Requested sizes are rounded up to the nearest **2 MiB boundary** (or
//! power-of-two for sizes < 2 MiB).  This eliminates fragmentation caused
//! by small variations in tensor layout or model padding.
//!
//! ## Zero-free guarantee
//!
//! During normal operation, `recycle()` **always** accepts the buffer —
//! memory is never released back to the driver.  The per-bucket capacity
//! limit is generous (32 buffers per bucket, configurable).  Only at
//! engine shutdown does `drain()` release all pooled memory.
//!
//! # VRAM accounting
//!
//! Every device allocation made through [`GpuContext::alloc`] is tracked via
//! atomic counters.  [`vram_usage`] returns `(current, peak)` without locking.
//!
//! Accounting contract:
//! - **Fresh allocation** (pool miss): `current += bucket_size`, `peak = max(peak, current)`.
//! - **Pool hit**: no change (bytes were already counted).
//! - **Recycle accepted**: no change (bytes remain allocated on device).
//! - **Recycle overflow** (bucket full): `current -= buf.len()` (buffer freed).
//! - **Pool drain**: `current -= sum(all pooled buffer sizes)`.
//!
//! Callers that drop a `CudaSlice` without calling [`recycle`] cause `current`
//! to remain elevated.  The pipeline MUST recycle consumed buffers to keep
//! accounting accurate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use tracing::{debug, info, warn};

use crate::error::{EngineError, Result};

/// Hardware-preferred alignment for tensor buffers (512 B — matches
/// NVIDIA L2 cache line and warp-coalesced access granularity).
pub const TENSOR_ALIGNMENT: usize = 512;

/// GPU memory alignment for DMA transfers (2 MiB — matches TLB large-page).
pub const DMA_ALIGNMENT: usize = 2 * 1024 * 1024;

// ─── VRAM accounting ─────────────────────────────────────────────────────────

/// Atomic VRAM byte counters.  Lock-free reads.
struct VramAccounting {
    current: AtomicUsize,
    peak: AtomicUsize,
}

impl VramAccounting {
    const fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn on_alloc(&self, bytes: usize) {
        let prev = self.current.fetch_add(bytes, Ordering::Relaxed);
        let new = prev + bytes;
        // CAS loop to update peak (Relaxed is fine — advisory metric).
        self.peak.fetch_max(new, Ordering::Relaxed);
    }

    #[inline]
    fn on_free(&self, bytes: usize) {
        self.current.fetch_sub(bytes, Ordering::Relaxed);
    }

    #[inline]
    fn snapshot(&self) -> (usize, usize) {
        (
            self.current.load(Ordering::Relaxed),
            self.peak.load(Ordering::Relaxed),
        )
    }
}

// ─── GPU context ─────────────────────────────────────────────────────────────

/// Long-lived GPU context shared across the entire engine.
pub struct GpuContext {
    device: Arc<CudaDevice>,

    /// Dedicated stream for video decode (NVDEC output DMA).
    pub decode_stream: CudaStream,

    /// Dedicated stream for preprocessing GPU kernels.
    pub preprocess_stream: CudaStream,

    /// Dedicated stream for inference execution.
    pub inference_stream: CudaStream,

    /// Bucketed buffer pool — zero-free steady state.
    buffer_pool: Mutex<BucketedPool>,

    /// Pool statistics (lock-free).
    pub pool_stats: PoolStats,

    /// Device memory accounting (lock-free reads).
    vram: VramAccounting,

    /// Allocation policy (warm-up vs steady state).
    pub alloc_policy: AllocPolicy,

    /// GPU performance profiler.
    pub profiler: PerfProfiler,

    /// Queue depth tracker for backpressure observability.
    pub queue_depth: QueueDepthTracker,

    /// VRAM limit (bytes). 0 = unlimited.
    vram_limit: AtomicUsize,
}

impl GpuContext {
    /// Initialize the GPU context on the given device ordinal.
    pub fn new(device_ordinal: usize) -> Result<Arc<Self>> {
        let device = CudaDevice::new(device_ordinal)?;

        let decode_stream = device.fork_default_stream()?;
        let preprocess_stream = device.fork_default_stream()?;
        let inference_stream = device.fork_default_stream()?;

        Ok(Arc::new(Self {
            device,
            decode_stream,
            preprocess_stream,
            inference_stream,
            buffer_pool: Mutex::new(BucketedPool::new()),
            pool_stats: PoolStats::new(),
            vram: VramAccounting::new(),
            alloc_policy: AllocPolicy::new(),
            profiler: PerfProfiler::new(),
            queue_depth: QueueDepthTracker::new(),
            vram_limit: AtomicUsize::new(0),
        }))
    }

    /// Access the underlying `CudaDevice`.
    #[inline]
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Allocate `size` bytes of device memory, preferring a pooled buffer.
    ///
    /// The returned buffer may be **larger** than `size` due to bucket
    /// alignment.  Callers must use only the first `size` bytes but pass
    /// the full buffer back to [`recycle`] to maintain pool consistency.
    ///
    /// VRAM accounting: incremented by `bucket_size` on fresh allocation (pool miss).
    pub fn alloc(&self, size: usize) -> Result<CudaSlice<u8>> {
        let bucket_size = bucket_for(size);

        // Try pool first — no accounting change (bytes already tracked).
        {
            let mut pool = self.buffer_pool.lock().unwrap();
            if let Some(buf) = pool.take(bucket_size) {
                self.pool_stats.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(buf);
            }
        }

        // Pool miss — allocate from driver at bucket-aligned size.
        self.pool_stats.misses.fetch_add(1, Ordering::Relaxed);
        let buf = self.device.alloc_zeros::<u8>(bucket_size)?;
        self.vram.on_alloc(bucket_size);

        if self.alloc_policy.is_steady_state() {
            warn!(
                bucket_size,
                "Pool miss in steady state — pool may be undersized"
            );
        }

        // Phase 9: VRAM limit enforcement.
        let limit = self.vram_limit.load(Ordering::Relaxed);
        if limit > 0 {
            let (current, _) = self.vram.snapshot();
            if current > limit {
                warn!(
                    current_mb = current / (1024 * 1024),
                    limit_mb = limit / (1024 * 1024),
                    "VRAM usage exceeds configured limit"
                );
            }
        }

        Ok(buf)
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// In the zero-free steady state, buffers are **never freed** to the
    /// driver — they are stored in the appropriate bucket for reuse.
    ///
    /// If the bucket is at capacity (overflow), the buffer is freed and
    /// VRAM accounting decremented.  This should only occur during warm-up
    /// or unusual workloads.
    pub fn recycle(&self, buf: CudaSlice<u8>) {
        let actual_size = buf.len();
        let mut pool = self.buffer_pool.lock().unwrap();
        if let Some(rejected) = pool.put(buf) {
            // Bucket overflow — free to driver and decrement accounting.
            self.pool_stats.overflows.fetch_add(1, Ordering::Relaxed);
            self.vram.on_free(actual_size);
            drop(rejected);
        } else {
            self.pool_stats.recycled.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Read current and peak VRAM usage (bytes) for allocations through this context.
    ///
    /// Lock-free.  Values are advisory and may lag slightly under contention.
    #[inline]
    pub fn vram_usage(&self) -> (usize, usize) {
        self.vram.snapshot()
    }

    /// Manually decrement VRAM accounting by `bytes`.
    ///
    /// Used when device memory allocated through this context is freed
    /// externally (e.g., `OutputRing` slot drop without going through `recycle`).
    #[inline]
    pub fn vram_dec(&self, bytes: usize) {
        self.vram.on_free(bytes);
    }

    /// Synchronize a specific stream, blocking until all enqueued work completes.
    pub fn sync_stream(stream: &CudaStream) -> Result<()> {
        stream.synchronize()?;
        Ok(())
    }

    /// Synchronize all three engine streams.
    pub fn sync_all(&self) -> Result<()> {
        self.decode_stream.synchronize()?;
        self.preprocess_stream.synchronize()?;
        self.inference_stream.synchronize()?;
        Ok(())
    }

    /// Report pool and VRAM statistics.
    pub fn report_pool_stats(&self) {
        let stats = &self.pool_stats;
        let pool = self.buffer_pool.lock().unwrap();
        let (vram_current, vram_peak) = self.vram.snapshot();

        info!(
            hits = stats.hits.load(Ordering::Relaxed),
            misses = stats.misses.load(Ordering::Relaxed),
            recycled = stats.recycled.load(Ordering::Relaxed),
            overflows = stats.overflows.load(Ordering::Relaxed),
            total_pooled = pool.total_buffers(),
            pooled_bytes_mb = pool.total_bytes() / (1024 * 1024),
            buckets = pool.bucket_count(),
            vram_current_mb = vram_current / (1024 * 1024),
            vram_peak_mb = vram_peak / (1024 * 1024),
            "Buffer pool report"
        );

        // Report profiler stats.
        self.profiler.report();
    }

    /// Set a VRAM usage cap (bytes).  0 = unlimited.
    ///
    /// Allocations that push `current` above this cap emit a tracing warning.
    /// The allocation still succeeds — the cap is advisory, not a hard limit,
    /// to avoid stalling the pipeline.
    pub fn set_vram_limit(&self, limit_bytes: usize) {
        self.vram_limit.store(limit_bytes, Ordering::Relaxed);
        info!(limit_mb = limit_bytes / (1024 * 1024), "VRAM limit set");
    }

    /// Capture a structured health snapshot for telemetry export.
    pub fn health_snapshot(&self) -> HealthSnapshot {
        let (vram_current, vram_peak) = self.vram.snapshot();
        let stats = &self.pool_stats;
        HealthSnapshot {
            vram_current_bytes: vram_current,
            vram_peak_bytes: vram_peak,
            vram_limit_bytes: self.vram_limit.load(Ordering::Relaxed),
            pool_hits: stats.hits.load(Ordering::Relaxed),
            pool_misses: stats.misses.load(Ordering::Relaxed),
            pool_hit_rate: stats.hit_rate(),
            pool_overflows: stats.overflows.load(Ordering::Relaxed),
            steady_state: self.alloc_policy.is_steady_state(),
            decode_queue_depth: self.queue_depth.decode.load(Ordering::Relaxed),
            preprocess_queue_depth: self.queue_depth.preprocess.load(Ordering::Relaxed),
            inference_queue_depth: self.queue_depth.inference.load(Ordering::Relaxed),
        }
    }
    /// Allocate device memory with explicit byte alignment.
    ///
    /// Rounds `size` up to the nearest `alignment` boundary, then routes
    /// through the bucketed pool.  Use `TENSOR_ALIGNMENT` (512B) for
    /// coalesced tensor access or `DMA_ALIGNMENT` (2MiB) for page-aligned DMA.
    #[inline]
    pub fn alloc_aligned(&self, size: usize, alignment: usize) -> Result<CudaSlice<u8>> {
        let aligned = (size + alignment - 1) & !(alignment - 1);
        self.alloc(aligned)
    }

    /// Issue an async L2 prefetch hint for a device buffer on the given stream.
    ///
    /// This is a performance hint — the GPU prefetches `count` bytes from
    /// `device_ptr` into L2 cache on the given stream.  No-op if the driver
    /// does not support `cuMemPrefetchAsync`.
    pub fn prefetch_l2(&self, device_ptr: u64, count: usize, stream: &CudaStream) -> Result<()> {
        extern "C" {
            fn cuMemPrefetchAsync(
                devPtr: u64,
                count: usize,
                dstDevice: i32,
                hStream: crate::codecs::sys::CUstream,
            ) -> crate::codecs::sys::CUresult;
        }
        let raw_stream = crate::codecs::nvdec::get_raw_stream(stream);
        // device ID 0 — single GPU context.
        unsafe {
            let r = cuMemPrefetchAsync(device_ptr, count, 0, raw_stream);
            if r != 0 {
                debug!(cu_result = r, "cuMemPrefetchAsync hint ignored (non-fatal)");
            }
        }
        Ok(())
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if let Ok(mut pool) = self.buffer_pool.lock() {
            let freed = pool.drain();
            self.vram.on_free(freed);
        }
    }
}

// ─── Pool statistics ────────────────────────────────────────────────────────

/// Lock-free pool access counters.
pub struct PoolStats {
    /// Number of `alloc()` calls satisfied from the pool.
    pub hits: AtomicU64,
    /// Number of `alloc()` calls that required a driver allocation.
    pub misses: AtomicU64,
    /// Number of `recycle()` calls that successfully returned a buffer.
    pub recycled: AtomicU64,
    /// Number of `recycle()` calls where the bucket was full (buffer freed).
    pub overflows: AtomicU64,
}

impl PoolStats {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            recycled: AtomicU64::new(0),
            overflows: AtomicU64::new(0),
        }
    }

    /// Hit rate as a percentage (0.0–100.0).
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        let total = h + m;
        if total == 0.0 {
            0.0
        } else {
            (h / total) * 100.0
        }
    }
}

// ─── Allocation policy ──────────────────────────────────────────────────────

/// Tracks warm-up vs steady-state allocation mode.
///
/// During warm-up, pool misses are expected (populating buckets).
/// After `enter_steady_state()`, misses trigger warnings — they indicate
/// the pool is undersized or the workload changed.
pub struct AllocPolicy {
    steady: AtomicBool,
}

impl AllocPolicy {
    fn new() -> Self {
        Self {
            steady: AtomicBool::new(false),
        }
    }

    /// Mark the pool as fully warmed — subsequent misses are anomalous.
    pub fn enter_steady_state(&self) {
        self.steady.store(true, Ordering::Release);
        info!("AllocPolicy: entered steady state — pool misses are now warnings");
    }

    /// Reset to warm-up mode.
    pub fn reset(&self) {
        self.steady.store(false, Ordering::Release);
    }

    #[inline]
    pub fn is_steady_state(&self) -> bool {
        self.steady.load(Ordering::Acquire)
    }
}

// ─── GPU performance profiler ───────────────────────────────────────────────

/// Lightweight per-stage GPU timing profiler.
///
/// Records wall-clock stage durations and kernel launch overhead.
/// Thread-safe via atomic accumulators — no locks in the hot path.
pub struct PerfProfiler {
    /// Preprocess kernel time (μs).
    pub preprocess_gpu_us: AtomicU64,
    pub preprocess_count: AtomicU64,
    /// Inference time (μs).
    pub inference_gpu_us: AtomicU64,
    pub inference_count: AtomicU64,
    /// Postprocess kernel time (μs).
    pub postprocess_gpu_us: AtomicU64,
    pub postprocess_count: AtomicU64,
    /// Encode time (μs).
    pub encode_gpu_us: AtomicU64,
    pub encode_count: AtomicU64,
    /// Kernel launch overhead samples (μs) — time from CPU dispatch to GPU start.
    pub launch_overhead_us: AtomicU64,
    pub launch_overhead_count: AtomicU64,
    /// Peak single-frame pipeline latency (μs).
    pub peak_frame_us: AtomicU64,
}

impl PerfProfiler {
    fn new() -> Self {
        Self {
            preprocess_gpu_us: AtomicU64::new(0),
            preprocess_count: AtomicU64::new(0),
            inference_gpu_us: AtomicU64::new(0),
            inference_count: AtomicU64::new(0),
            postprocess_gpu_us: AtomicU64::new(0),
            postprocess_count: AtomicU64::new(0),
            encode_gpu_us: AtomicU64::new(0),
            encode_count: AtomicU64::new(0),
            launch_overhead_us: AtomicU64::new(0),
            launch_overhead_count: AtomicU64::new(0),
            peak_frame_us: AtomicU64::new(0),
        }
    }

    /// Record a stage timing sample.
    #[inline]
    pub fn record_stage(&self, stage: PerfStage, elapsed_us: u64) {
        let (acc, cnt) = match stage {
            PerfStage::Preprocess => (&self.preprocess_gpu_us, &self.preprocess_count),
            PerfStage::Inference => (&self.inference_gpu_us, &self.inference_count),
            PerfStage::Postprocess => (&self.postprocess_gpu_us, &self.postprocess_count),
            PerfStage::Encode => (&self.encode_gpu_us, &self.encode_count),
        };
        acc.fetch_add(elapsed_us, Ordering::Relaxed);
        cnt.fetch_add(1, Ordering::Relaxed);
    }

    /// Record kernel launch overhead.
    #[inline]
    pub fn record_launch_overhead(&self, overhead_us: u64) {
        self.launch_overhead_us
            .fetch_add(overhead_us, Ordering::Relaxed);
        self.launch_overhead_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record peak frame latency.
    #[inline]
    pub fn record_frame_latency(&self, latency_us: u64) {
        self.peak_frame_us.fetch_max(latency_us, Ordering::Relaxed);
    }

    /// Compute average for a stage.
    fn avg(total: &AtomicU64, count: &AtomicU64) -> u64 {
        let c = count.load(Ordering::Relaxed);
        if c > 0 {
            total.load(Ordering::Relaxed) / c
        } else {
            0
        }
    }

    /// Report profiling results.
    pub fn report(&self) {
        info!(
            preprocess_avg_us = Self::avg(&self.preprocess_gpu_us, &self.preprocess_count),
            inference_avg_us = Self::avg(&self.inference_gpu_us, &self.inference_count),
            postprocess_avg_us = Self::avg(&self.postprocess_gpu_us, &self.postprocess_count),
            encode_avg_us = Self::avg(&self.encode_gpu_us, &self.encode_count),
            launch_overhead_avg_us =
                Self::avg(&self.launch_overhead_us, &self.launch_overhead_count),
            peak_frame_us = self.peak_frame_us.load(Ordering::Relaxed),
            "GPU profiler report"
        );
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.preprocess_gpu_us.store(0, Ordering::Relaxed);
        self.preprocess_count.store(0, Ordering::Relaxed);
        self.inference_gpu_us.store(0, Ordering::Relaxed);
        self.inference_count.store(0, Ordering::Relaxed);
        self.postprocess_gpu_us.store(0, Ordering::Relaxed);
        self.postprocess_count.store(0, Ordering::Relaxed);
        self.encode_gpu_us.store(0, Ordering::Relaxed);
        self.encode_count.store(0, Ordering::Relaxed);
        self.launch_overhead_us.store(0, Ordering::Relaxed);
        self.launch_overhead_count.store(0, Ordering::Relaxed);
        self.peak_frame_us.store(0, Ordering::Relaxed);
    }
}

/// Pipeline stages for profiling.
#[derive(Clone, Copy, Debug)]
pub enum PerfStage {
    Preprocess,
    Inference,
    Postprocess,
    Encode,
}

// ─── Bucketed buffer pool ───────────────────────────────────────────────────

/// Bucket alignment: 2 MiB.  All allocations are rounded up to this boundary.
/// This eliminates fragmentation from small size variations.
const BUCKET_ALIGNMENT: usize = 2 * 1024 * 1024; // 2 MiB

/// Maximum buffers per bucket before overflow (freed to driver).
/// At 2 MiB alignment:
/// - 32 buffers × 2 MiB = 64 MiB minimum per active bucket
/// - Typical pipeline uses 3–5 buckets ≈ 200–320 MiB pooled
const MAX_PER_BUCKET: usize = 32;

/// Power-of-two bucketed buffer pool.
///
/// Strategy:
/// 1. Requested size is rounded up to `BUCKET_ALIGNMENT` (2 MiB).
/// 2. Each bucket holds a LIFO stack of buffers at that size.
/// 3. `take()` pops from the stack — O(1).
/// 4. `put()` pushes back — O(1).
/// 5. Overflow only occurs if a single bucket exceeds `MAX_PER_BUCKET`.
///
/// After warm-up (first pass through the pipeline), the pool holds enough
/// buffers for sustained zero-allocation operation.
struct BucketedPool {
    /// Buckets keyed by aligned size.  Each value is a LIFO stack.
    buckets: HashMap<usize, Vec<CudaSlice<u8>>>,
}

impl BucketedPool {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Try to pop a buffer from the bucket matching `bucket_size`.
    fn take(&mut self, bucket_size: usize) -> Option<CudaSlice<u8>> {
        let stack = self.buckets.get_mut(&bucket_size)?;
        let buf = stack.pop()?;
        if stack.is_empty() {
            self.buckets.remove(&bucket_size);
        }
        Some(buf)
    }

    /// Push a buffer into its bucket.
    ///
    /// Returns `Some(buf)` if the bucket is at capacity (caller must free).
    fn put(&mut self, buf: CudaSlice<u8>) -> Option<CudaSlice<u8>> {
        let bucket_size = buf.len();
        let stack = self.buckets.entry(bucket_size).or_default();
        if stack.len() >= MAX_PER_BUCKET {
            return Some(buf);
        }
        stack.push(buf);
        None
    }

    /// Total number of buffers across all buckets.
    fn total_buffers(&self) -> usize {
        self.buckets.values().map(|s| s.len()).sum()
    }

    /// Total bytes held in the pool.
    fn total_bytes(&self) -> usize {
        self.buckets
            .iter()
            .map(|(size, stack)| size * stack.len())
            .sum()
    }

    /// Number of active buckets.
    fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Release all pooled buffers.  Returns total bytes freed.
    fn drain(&mut self) -> usize {
        let mut freed = 0usize;
        for (size, stack) in self.buckets.drain() {
            let count = stack.len();
            freed += size * count;
            for buf in stack {
                drop(buf);
            }
        }
        freed
    }
}

// ─── Bucket sizing ──────────────────────────────────────────────────────────

/// Round `size` up to the nearest bucket boundary.
///
/// For sizes < 2 MiB: round up to next power of two (minimum 4096 bytes).
/// For sizes ≥ 2 MiB: round up to next 2 MiB boundary.
///
/// This ensures that small allocations (kernel temporaries) use tight
/// power-of-two buckets, while large allocations (frame buffers, tensors)
/// use 2 MiB-aligned buckets that absorb padding variations.
#[inline]
fn bucket_for(size: usize) -> usize {
    if size == 0 {
        return BUCKET_ALIGNMENT.min(4096);
    }
    if size < BUCKET_ALIGNMENT {
        // Power-of-two, minimum 4096 (one page).
        size.max(4096).next_power_of_two()
    } else {
        // Round up to next 2 MiB boundary.
        (size + BUCKET_ALIGNMENT - 1) & !(BUCKET_ALIGNMENT - 1)
    }
}

// ─── Stream overlap timing (Phase 7 — concurrency audit) ───────────────────

/// Measures overlap between stages running on different CUDA streams.
///
/// Records a `decode_done` event on the decode stream and an
/// `inference_start` event on the inference stream.  After sync,
/// `cuEventElapsedTime(decode_done, inference_start)` yields the gap (ms):
/// - **Negative** → streams overlapped (concurrent execution proven).
/// - **Zero** → back-to-back (no gap, no overlap).
/// - **Positive** → pipeline bubble (inference waited for decode).
pub struct StreamOverlapTimer {
    decode_done: crate::codecs::sys::CUevent,
    inference_start: crate::codecs::sys::CUevent,
    samples: Mutex<Vec<f32>>,
}

impl StreamOverlapTimer {
    pub fn new() -> Result<Self> {
        use crate::codecs::sys;
        let mut d: sys::CUevent = std::ptr::null_mut();
        let mut i: sys::CUevent = std::ptr::null_mut();
        unsafe {
            sys::check_cu(sys::cuEventCreate(&mut d, 0), "overlap decode_done create")?;
            sys::check_cu(sys::cuEventCreate(&mut i, 0), "overlap infer_start create")?;
        }
        Ok(Self {
            decode_done: d,
            inference_start: i,
            samples: Mutex::new(Vec::with_capacity(1024)),
        })
    }

    /// Record decode-done event on the decode stream.
    pub fn mark_decode_done(&self, stream: &CudaStream) -> Result<()> {
        let raw = crate::codecs::nvdec::get_raw_stream(stream);
        unsafe {
            crate::codecs::sys::check_cu(
                crate::codecs::sys::cuEventRecord(self.decode_done, raw),
                "overlap decode_done record",
            )
        }
    }

    /// Record inference-start event on the inference stream.
    pub fn mark_inference_start(&self, stream: &CudaStream) -> Result<()> {
        let raw = crate::codecs::nvdec::get_raw_stream(stream);
        unsafe {
            crate::codecs::sys::check_cu(
                crate::codecs::sys::cuEventRecord(self.inference_start, raw),
                "overlap infer_start record",
            )
        }
    }

    /// Compute elapsed time between decode_done and inference_start.
    /// Both events must have completed (streams synchronized).
    /// Stores the sample internally.
    pub fn sample(&self) -> Result<f32> {
        let mut ms: f32 = 0.0;
        extern "C" {
            fn cuEventElapsedTime(
                pMilliseconds: *mut f32,
                hStart: crate::codecs::sys::CUevent,
                hEnd: crate::codecs::sys::CUevent,
            ) -> crate::codecs::sys::CUresult;
        }
        unsafe {
            crate::codecs::sys::check_cu(
                cuEventElapsedTime(&mut ms, self.decode_done, self.inference_start),
                "overlap elapsed",
            )?;
        }
        self.samples.lock().unwrap().push(ms);
        Ok(ms)
    }

    /// Compute overlap statistics from all collected samples.
    pub fn report(&self) -> OverlapReport {
        let samples = self.samples.lock().unwrap();
        if samples.is_empty() {
            return OverlapReport {
                sample_count: 0,
                avg_gap_ms: 0.0,
                min_gap_ms: 0.0,
                max_gap_ms: 0.0,
                overlap_count: 0,
                overlap_pct: 0.0,
            };
        }
        let n = samples.len();
        let sum: f32 = samples.iter().sum();
        let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let overlaps = samples.iter().filter(|&&g| g < 0.0).count();
        OverlapReport {
            sample_count: n,
            avg_gap_ms: sum / n as f32,
            min_gap_ms: min,
            max_gap_ms: max,
            overlap_count: overlaps,
            overlap_pct: (overlaps as f64 / n as f64) * 100.0,
        }
    }
}

impl Drop for StreamOverlapTimer {
    fn drop(&mut self) {
        unsafe {
            crate::codecs::sys::cuEventDestroy_v2(self.decode_done);
            crate::codecs::sys::cuEventDestroy_v2(self.inference_start);
        }
    }
}

/// Summary of stream overlap measurements.
#[derive(Debug, Clone)]
pub struct OverlapReport {
    pub sample_count: usize,
    /// Average gap in ms (negative = overlap).
    pub avg_gap_ms: f32,
    pub min_gap_ms: f32,
    pub max_gap_ms: f32,
    /// Number of samples where streams overlapped.
    pub overlap_count: usize,
    /// Percentage of samples with overlap.
    pub overlap_pct: f64,
}

// ─── Queue Depth Tracking ───────────────────────────────────────────────────

/// Lock-free queue depth counters for backpressure observability.
pub struct QueueDepthTracker {
    pub decode: AtomicUsize,
    pub preprocess: AtomicUsize,
    pub inference: AtomicUsize,
}

impl QueueDepthTracker {
    pub fn new() -> Self {
        Self {
            decode: AtomicUsize::new(0),
            preprocess: AtomicUsize::new(0),
            inference: AtomicUsize::new(0),
        }
    }
}

// ─── Health Snapshot ────────────────────────────────────────────────────────

/// Immutable snapshot of engine health metrics for telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct HealthSnapshot {
    pub vram_current_bytes: usize,
    pub vram_peak_bytes: usize,
    pub vram_limit_bytes: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub pool_hit_rate: f64,
    pub pool_overflows: usize,
    pub steady_state: bool,
    pub decode_queue_depth: usize,
    pub preprocess_queue_depth: usize,
    pub inference_queue_depth: usize,
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_sizing_small() {
        assert_eq!(bucket_for(1), 4096);
        assert_eq!(bucket_for(4096), 4096);
        assert_eq!(bucket_for(4097), 8192);
        assert_eq!(bucket_for(1_000_000), 1_048_576); // 1 MiB
        assert_eq!(bucket_for(1_048_576), 1_048_576);
    }

    #[test]
    fn bucket_sizing_large() {
        let two_mb = 2 * 1024 * 1024;
        assert_eq!(bucket_for(two_mb), two_mb);
        assert_eq!(bucket_for(two_mb + 1), two_mb * 2);
        assert_eq!(bucket_for(5_000_000), 3 * two_mb); // 6 MiB
        assert_eq!(bucket_for(10_000_000), 5 * two_mb); // 10 MiB
    }

    #[test]
    fn bucket_sizing_zero() {
        assert!(bucket_for(0) > 0);
    }
}
