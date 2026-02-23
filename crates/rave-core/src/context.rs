//! Shared CUDA context — single device, explicit stream management,
//! bucketed buffer pool, VRAM accounting, hardware-aligned allocation,
//! GPU profiling hooks, and production metrics enforcement.
//!
//! # Buffer pool (zero-free steady state)
//!
//! All device allocations go through [`GpuContext::alloc`] and recycled
//! buffers are returned via [`GpuContext::recycle`].  After warm-up, the
//! pool holds enough buffers to satisfy every frame without hitting the
//! CUDA driver allocator.

use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DeviceSlice};
use tracing::{info, warn};

use crate::error::{EngineError, Result};
use crate::ffi_types::{CUDA_SUCCESS, CUresult, CUstream};

#[cfg(not(target_os = "linux"))]
unsafe extern "C" {
    fn cuInit(flags: u32) -> CUresult;
    fn cuDriverGetVersion(driver_version: *mut i32) -> CUresult;
    fn cuDeviceGetCount(count: *mut i32) -> CUresult;
    fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dladdr(addr: *const c_void, info: *mut DlInfo) -> i32;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "linux")]
const RTLD_NOW: i32 = 2;
#[cfg(target_os = "linux")]
const RTLD_GLOBAL: i32 = 0x100;
#[cfg(target_os = "linux")]
const RTLD_LOCAL: i32 = 0;

#[cfg(target_os = "linux")]
#[repr(C)]
struct DlInfo {
    dli_fname: *const c_char,
    dli_fbase: *mut c_void,
    dli_sname: *const c_char,
    dli_saddr: *mut c_void,
}

static CUDA_INIT_DIAG_ONCE: Once = Once::new();

#[cfg(target_os = "linux")]
struct CudaDriverSymbols {
    cu_init: unsafe extern "C" fn(u32) -> CUresult,
    cu_driver_get_version: unsafe extern "C" fn(*mut i32) -> CUresult,
    cu_device_get_count: unsafe extern "C" fn(*mut i32) -> CUresult,
    cu_stream_synchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    cu_init_addr: usize,
}

#[cfg(target_os = "linux")]
static CUDA_DRIVER_SYMBOLS: OnceLock<std::result::Result<CudaDriverSymbols, String>> =
    OnceLock::new();

#[cfg(target_os = "linux")]
fn load_cuda_symbol<T>(handle: *mut c_void, name: &'static str) -> std::result::Result<T, String> {
    let cname = CString::new(name).map_err(|_| format!("invalid CUDA symbol name: {name}"))?;
    // SAFETY: handle is a dlopen handle and cname is a valid NUL-terminated symbol name.
    let ptr = unsafe { dlsym(handle, cname.as_ptr()) };
    if ptr.is_null() {
        let err = unsafe {
            let p = dlerror();
            if p.is_null() {
                "unknown dlsym error".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().to_string()
            }
        };
        Err(format!("dlsym({name}) failed: {err}"))
    } else {
        // SAFETY: ptr points to a function symbol with signature T.
        Ok(unsafe { std::mem::transmute_copy(&ptr) })
    }
}

#[cfg(target_os = "linux")]
fn init_cuda_driver_symbols() -> std::result::Result<CudaDriverSymbols, String> {
    let mut handle = std::ptr::null_mut();
    let mut last_err = "unknown dlopen error".to_string();
    for candidate in ["libcuda.so.1", "libcuda.so"] {
        let soname =
            CString::new(candidate).map_err(|_| format!("invalid CUDA soname: {candidate}"))?;
        // SAFETY: static soname and valid flags.
        handle = unsafe { dlopen(soname.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if !handle.is_null() {
            break;
        }
        last_err = unsafe {
            let p = dlerror();
            if p.is_null() {
                "unknown dlopen error".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().to_string()
            }
        };
    }

    if handle.is_null() {
        return Err(format!(
            "dlopen(libcuda.so.1|libcuda.so) failed: {last_err}"
        ));
    }

    let cu_init = load_cuda_symbol(handle, "cuInit")?;
    let cu_driver_get_version = load_cuda_symbol(handle, "cuDriverGetVersion")?;
    let cu_device_get_count = load_cuda_symbol(handle, "cuDeviceGetCount")?;
    let cu_stream_synchronize = load_cuda_symbol(handle, "cuStreamSynchronize")?;
    let cu_init_addr = {
        let fptr: unsafe extern "C" fn(u32) -> CUresult = cu_init;
        fptr as usize
    };
    Ok(CudaDriverSymbols {
        cu_init,
        cu_driver_get_version,
        cu_device_get_count,
        cu_stream_synchronize,
        cu_init_addr,
    })
}

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

    /// Enforce VRAM limit as a hard allocation error instead of warn-only.
    strict_vram_limit: AtomicBool,
}

impl GpuContext {
    #[cfg(target_os = "linux")]
    fn cuda_driver_symbols() -> Result<&'static CudaDriverSymbols> {
        let symbols = CUDA_DRIVER_SYMBOLS.get_or_init(init_cuda_driver_symbols);
        symbols.as_ref().map_err(|err| {
            EngineError::Pipeline(format!(
                "failed to load CUDA driver symbols: {err}. \
Ensure NVIDIA driver libraries are installed and visible via LD_LIBRARY_PATH \
(on WSL, prepend /usr/lib/wsl/lib)."
            ))
        })
    }

    #[cfg(target_os = "linux")]
    fn cu_init(flags: u32) -> Result<CUresult> {
        let symbols = Self::cuda_driver_symbols()?;
        // SAFETY: symbol resolved from libcuda with matching signature.
        Ok(unsafe { (symbols.cu_init)(flags) })
    }

    #[cfg(not(target_os = "linux"))]
    fn cu_init(flags: u32) -> Result<CUresult> {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuInit(flags) })
    }

    #[cfg(target_os = "linux")]
    fn cu_driver_get_version(driver_version: &mut i32) -> Result<CUresult> {
        let symbols = Self::cuda_driver_symbols()?;
        // SAFETY: symbol resolved from libcuda with matching signature.
        Ok(unsafe { (symbols.cu_driver_get_version)(driver_version as *mut i32) })
    }

    #[cfg(not(target_os = "linux"))]
    fn cu_driver_get_version(driver_version: &mut i32) -> Result<CUresult> {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuDriverGetVersion(driver_version as *mut i32) })
    }

    #[cfg(target_os = "linux")]
    fn cu_device_get_count(count: &mut i32) -> Result<CUresult> {
        let symbols = Self::cuda_driver_symbols()?;
        // SAFETY: symbol resolved from libcuda with matching signature.
        Ok(unsafe { (symbols.cu_device_get_count)(count as *mut i32) })
    }

    #[cfg(not(target_os = "linux"))]
    fn cu_device_get_count(count: &mut i32) -> Result<CUresult> {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuDeviceGetCount(count as *mut i32) })
    }

    #[cfg(target_os = "linux")]
    fn cu_stream_synchronize(stream: CUstream) -> Result<CUresult> {
        let symbols = Self::cuda_driver_symbols()?;
        // SAFETY: symbol resolved from libcuda with matching signature.
        Ok(unsafe { (symbols.cu_stream_synchronize)(stream) })
    }

    #[cfg(not(target_os = "linux"))]
    fn cu_stream_synchronize(stream: CUstream) -> Result<CUresult> {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuStreamSynchronize(stream) })
    }

    #[cfg(target_os = "linux")]
    fn is_wsl2() -> bool {
        std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|s| s.to_ascii_lowercase().contains("microsoft"))
            .unwrap_or(false)
    }

    #[cfg(not(target_os = "linux"))]
    #[allow(dead_code)]
    fn is_wsl2() -> bool {
        false
    }

    #[cfg(target_os = "linux")]
    fn loaded_shared_object_path(needle: &str) -> Option<String> {
        let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
        maps.lines()
            .filter_map(|line| line.split_whitespace().last())
            .find(|p| p.contains(needle))
            .map(|s| s.to_string())
    }

    #[cfg(target_os = "linux")]
    fn loaded_libcuda_path() -> Option<String> {
        Self::loaded_shared_object_path("libcuda.so")
    }

    #[cfg(target_os = "linux")]
    fn loaded_libnvidia_encode_path() -> Option<String> {
        Self::loaded_shared_object_path("libnvidia-encode.so")
    }

    #[cfg(target_os = "linux")]
    fn libcuda_path_from_dladdr() -> Option<String> {
        let cu_init_addr = Self::cuda_driver_symbols().ok()?.cu_init_addr as *const c_void;
        let mut info = DlInfo {
            dli_fname: std::ptr::null(),
            dli_fbase: std::ptr::null_mut(),
            dli_sname: std::ptr::null(),
            dli_saddr: std::ptr::null_mut(),
        };
        // SAFETY: passing symbol pointer address and valid DlInfo out pointer.
        let rc = unsafe { dladdr(cu_init_addr, &mut info as *mut DlInfo) };
        if rc == 0 || info.dli_fname.is_null() {
            None
        } else {
            // SAFETY: dli_fname is a valid C string on success.
            Some(
                unsafe { CStr::from_ptr(info.dli_fname) }
                    .to_string_lossy()
                    .to_string(),
            )
        }
    }

    #[cfg(target_os = "linux")]
    fn preload_wsl_libcuda() -> Result<()> {
        if !Self::is_wsl2() {
            return Ok(());
        }
        let lib = CString::new("/usr/lib/wsl/lib/libcuda.so.1")
            .map_err(|_| crate::error::EngineError::Pipeline("invalid WSL libcuda path".into()))?;
        // SAFETY: static NUL-terminated path, valid dlopen flags.
        let handle = unsafe { dlopen(lib.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle.is_null() {
            // SAFETY: dlerror returns thread-local string or null.
            let err = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().to_string()
                }
            };
            return Err(crate::error::EngineError::Pipeline(format!(
                "Failed to load WSL CUDA driver '/usr/lib/wsl/lib/libcuda.so.1' ({err}). \
WSL requires NVIDIA driver libs from /usr/lib/wsl/lib."
            )));
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn preload_wsl_libcuda() -> Result<()> {
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn preload_wsl_nvenc_for_diagnostics() {
        if !Self::is_wsl2() {
            return;
        }
        let soname =
            CString::new("libnvidia-encode.so.1").expect("static libnvidia-encode soname is valid");
        // SAFETY: static soname and valid flags.
        let handle = unsafe { dlopen(soname.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            // SAFETY: dlerror returns thread-local string or null.
            let err = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().to_string()
                }
            };
            warn!(
                error = %err,
                "WSL loader could not open libnvidia-encode.so.1; skip-encode benchmarks can still run"
            );
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn preload_wsl_nvenc_for_diagnostics() {}

    #[cfg(target_os = "linux")]
    fn try_nvml_preinit_for_wsl() -> std::result::Result<(), String> {
        type NvmlInitV2 = unsafe extern "C" fn() -> i32;
        let lib =
            CString::new("libnvidia-ml.so.1").map_err(|_| "invalid NVML soname".to_string())?;
        // SAFETY: static soname and valid flags.
        let handle = unsafe { dlopen(lib.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle.is_null() {
            let err = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().to_string()
                }
            };
            return Err(format!("dlopen(libnvidia-ml.so.1) failed: {err}"));
        }

        let sym = CString::new("nvmlInit_v2").map_err(|_| "invalid NVML symbol".to_string())?;
        // SAFETY: handle is valid and symbol name is NUL-terminated.
        let fptr = unsafe { dlsym(handle, sym.as_ptr()) };
        if fptr.is_null() {
            let err = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "unknown dlsym error".to_string()
                } else {
                    CStr::from_ptr(p).to_string_lossy().to_string()
                }
            };
            return Err(format!("dlsym(nvmlInit_v2) failed: {err}"));
        }

        // SAFETY: symbol address came from dlsym for nvmlInit_v2.
        let nvml_init: NvmlInitV2 = unsafe { std::mem::transmute(fptr) };
        let rc = unsafe { nvml_init() };
        if rc != 0 {
            return Err(format!("nvmlInit_v2 returned rc={rc}"));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn ensure_cuda_init_for_wsl() -> Result<()> {
        if !Self::is_wsl2() {
            return Ok(());
        }

        let rc = Self::cu_init(0)?;
        if rc == CUDA_SUCCESS {
            return Ok(());
        }

        if rc != 304 {
            return Err(crate::error::EngineError::Pipeline(format!(
                "cuInit failed on WSL with rc={rc}; run scripts/wsl_gpu_healthcheck.sh"
            )));
        }

        let nvml_preinit = Self::try_nvml_preinit_for_wsl();
        let rc_retry = Self::cu_init(0)?;
        if rc_retry == CUDA_SUCCESS {
            warn!(
                nvml_preinit = ?nvml_preinit.as_ref().map(|_| "ok"),
                "Recovered cuInit(304) on WSL after NVML pre-init workaround"
            );
            return Ok(());
        }

        let nvml_msg = match nvml_preinit {
            Ok(()) => "nvmlInit_v2 ok".to_string(),
            Err(e) => e,
        };
        let libcuda_hint = Self::loaded_libcuda_path().unwrap_or_else(|| "unknown".to_string());
        let path_hint = if libcuda_hint.contains("/usr/lib/wsl/drivers/") {
            format!(
                " Detected libcuda from '{libcuda_hint}' (WSL driver package path); this commonly fails with undefined symbols such as cuPvtCompilePtx. \
Run with: LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}"
            )
        } else {
            String::new()
        };
        Err(crate::error::EngineError::Pipeline(format!(
            "cuInit failed with CUDA_ERROR_OPERATING_SYSTEM (304) on WSL (retry rc={rc_retry}; nvml_preinit={nvml_msg}). \
This is typically a WSL/Windows NVIDIA driver issue. Run scripts/wsl_gpu_healthcheck.sh.{path_hint}"
        )))
    }

    #[cfg(not(target_os = "linux"))]
    fn ensure_cuda_init_for_wsl() -> Result<()> {
        Ok(())
    }

    fn log_cuda_init_diagnostics_once() {
        CUDA_INIT_DIAG_ONCE.call_once(|| {
            let mut driver_version = -1i32;
            let rc_driver = match Self::cu_driver_get_version(&mut driver_version) {
                Ok(rc) => rc,
                Err(err) => {
                    warn!(error = %err, "CUDA diagnostics could not query driver version");
                    -1
                }
            };
            let mut device_count = -1i32;
            let rc_count = match Self::cu_device_get_count(&mut device_count) {
                Ok(rc) => rc,
                Err(err) => {
                    warn!(error = %err, "CUDA diagnostics could not query device count");
                    -1
                }
            };
            let rc_init = match Self::cu_init(0) {
                Ok(rc) => rc,
                Err(err) => {
                    warn!(error = %err, "CUDA diagnostics could not call cuInit");
                    -1
                }
            };
            #[cfg(target_os = "linux")]
            let libcuda_path = Self::loaded_libcuda_path();
            #[cfg(target_os = "linux")]
            let libcuda_dladdr_path = Self::libcuda_path_from_dladdr();
            #[cfg(target_os = "linux")]
            let libnvidia_encode_path = Self::loaded_libnvidia_encode_path();
            #[cfg(not(target_os = "linux"))]
            let libcuda_path: Option<String> = None;
            #[cfg(not(target_os = "linux"))]
            let libcuda_dladdr_path: Option<String> = None;
            #[cfg(not(target_os = "linux"))]
            let libnvidia_encode_path: Option<String> = None;

            info!(
                libcuda = ?libcuda_path,
                libcuda_from_dladdr = ?libcuda_dladdr_path,
                libnvidia_encode = ?libnvidia_encode_path,
                cu_init_rc = rc_init,
                cu_driver_get_version_rc = rc_driver,
                cu_driver_version = driver_version,
                cu_device_get_count_rc = rc_count,
                cu_device_count = device_count,
                "CUDA init diagnostics"
            );

            #[cfg(target_os = "linux")]
            if let Some(path) = &libcuda_path
                && path.contains("/usr/lib/wsl/drivers/")
            {
                warn!(
                    libcuda = %path,
                    "WSL loaded libcuda from /usr/lib/wsl/drivers; prepend LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}"
                );
            }
        });
    }

    #[cfg(target_os = "linux")]
    fn validate_no_cuda_stub() -> Result<()> {
        if let Some(path) = Self::loaded_libcuda_path()
            && path.contains("/usr/local/cuda")
            && path.contains("libcuda.so")
        {
            return Err(crate::error::EngineError::Pipeline(format!(
                "Detected CUDA toolkit stub driver at '{path}'. \
Use WSL NVIDIA driver libcuda instead: export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
            )));
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn validate_no_cuda_stub() -> Result<()> {
        Ok(())
    }

    /// Initialize the GPU context on the given device ordinal.
    pub fn new(device_ordinal: usize) -> Result<Arc<Self>> {
        Self::preload_wsl_libcuda()?;
        Self::preload_wsl_nvenc_for_diagnostics();
        Self::ensure_cuda_init_for_wsl()?;
        Self::log_cuda_init_diagnostics_once();
        Self::validate_no_cuda_stub()?;

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
            strict_vram_limit: AtomicBool::new(false),
        }))
    }

    /// Access the underlying `CudaDevice`.
    #[inline]
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Allocate `size` bytes of device memory, preferring a pooled buffer.
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
        let (current_before, _) = self.vram.snapshot();
        let would_exceed = check_vram_limit(
            self.strict_vram_limit.load(Ordering::Relaxed),
            match self.vram_limit.load(Ordering::Relaxed) {
                0 => None,
                limit => Some(limit),
            },
            current_before,
            size,
            bucket_size,
        )?;
        let buf = self.device.alloc_zeros::<u8>(bucket_size)?;
        self.vram.on_alloc(bucket_size);

        if self.alloc_policy.is_steady_state() {
            warn!(
                bucket_size,
                "Pool miss in steady state — pool may be undersized"
            );
        }

        if would_exceed {
            let limit = self.vram_limit.load(Ordering::Relaxed);
            let would_be = current_before.saturating_add(bucket_size);
            warn!(
                current_mb = current_before / (1024 * 1024),
                would_be_mb = would_be / (1024 * 1024),
                limit_mb = limit / (1024 * 1024),
                requested_bytes = size,
                reserved_bytes = bucket_size,
                "VRAM usage would exceed configured limit; continuing because strict_vram_limit=false"
            );
        }

        Ok(buf)
    }

    /// Return a buffer to the pool for future reuse.
    pub fn recycle(&self, buf: CudaSlice<u8>) {
        let actual_size = buf.len();
        let mut pool = self.buffer_pool.lock().unwrap();
        if let Some(rejected) = pool.put(buf) {
            self.pool_stats.overflows.fetch_add(1, Ordering::Relaxed);
            self.vram.on_free(actual_size);
            drop(rejected);
        } else {
            self.pool_stats.recycled.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Read current and peak VRAM usage (bytes) for allocations through this context.
    #[inline]
    pub fn vram_usage(&self) -> (usize, usize) {
        self.vram.snapshot()
    }

    /// Manually decrement VRAM accounting by `bytes`.
    #[inline]
    pub fn vram_dec(&self, bytes: usize) {
        self.vram.on_free(bytes);
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

        self.profiler.report();
    }

    /// Set a VRAM usage cap (bytes).  0 = unlimited.
    pub fn set_vram_limit(&self, limit_bytes: usize) {
        self.vram_limit.store(limit_bytes, Ordering::Relaxed);
        info!(limit_mb = limit_bytes / (1024 * 1024), "VRAM limit set");
    }

    /// Enable/disable hard-fail behavior when a configured VRAM limit would be exceeded.
    pub fn set_strict_vram_limit(&self, enabled: bool) {
        self.strict_vram_limit.store(enabled, Ordering::Relaxed);
        info!(enabled, "Strict VRAM limit enforcement");
    }

    /// Capture a structured health snapshot for telemetry export.
    pub fn health_snapshot(&self) -> HealthSnapshot {
        let (vram_current, vram_peak) = self.vram.snapshot();
        let stats = &self.pool_stats;
        HealthSnapshot {
            vram_current_bytes: vram_current,
            vram_peak_bytes: vram_peak,
            vram_limit_bytes: self.vram_limit.load(Ordering::Relaxed),
            pool_hits: stats.hits.load(Ordering::Relaxed) as usize,
            pool_misses: stats.misses.load(Ordering::Relaxed) as usize,
            pool_hit_rate: stats.hit_rate(),
            pool_overflows: stats.overflows.load(Ordering::Relaxed) as usize,
            steady_state: self.alloc_policy.is_steady_state(),
            decode_queue_depth: self.queue_depth.decode.load(Ordering::Relaxed),
            preprocess_queue_depth: self.queue_depth.preprocess.load(Ordering::Relaxed),
            inference_queue_depth: self.queue_depth.inference.load(Ordering::Relaxed),
        }
    }

    /// Allocate device memory with explicit byte alignment.
    #[inline]
    pub fn alloc_aligned(&self, size: usize, alignment: usize) -> Result<CudaSlice<u8>> {
        let aligned = (size + alignment - 1) & !(alignment - 1);
        self.alloc(aligned)
    }

    /// Synchronize a specific stream, blocking until all enqueued work completes.
    pub fn sync_stream(stream: &CudaStream) -> Result<()> {
        let raw = stream.stream as CUstream;
        let rc = Self::cu_stream_synchronize(raw)?;
        if rc == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(crate::error::EngineError::Decode(format!(
                "cuStreamSynchronize failed with CUDA error code {rc}"
            )))
        }
    }

    /// Synchronize all three engine streams.
    pub fn sync_all(&self) -> Result<()> {
        Self::sync_stream(&self.decode_stream)?;
        Self::sync_stream(&self.preprocess_stream)?;
        Self::sync_stream(&self.inference_stream)?;
        Ok(())
    }
}

// SAFETY: GpuContext manages thread-safe CUDA usage via internal synchronization/Mutex.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

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
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub recycled: AtomicU64,
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
pub struct AllocPolicy {
    steady: AtomicBool,
}

impl AllocPolicy {
    fn new() -> Self {
        Self {
            steady: AtomicBool::new(false),
        }
    }

    pub fn enter_steady_state(&self) {
        self.steady.store(true, Ordering::Release);
        info!("AllocPolicy: entered steady state — pool misses are now warnings");
    }

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
pub struct PerfProfiler {
    pub preprocess_gpu_us: AtomicU64,
    pub preprocess_count: AtomicU64,
    pub inference_gpu_us: AtomicU64,
    pub inference_count: AtomicU64,
    pub postprocess_gpu_us: AtomicU64,
    pub postprocess_count: AtomicU64,
    pub encode_gpu_us: AtomicU64,
    pub encode_count: AtomicU64,
    pub launch_overhead_us: AtomicU64,
    pub launch_overhead_count: AtomicU64,
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

    #[inline]
    pub fn record_launch_overhead(&self, overhead_us: u64) {
        self.launch_overhead_us
            .fetch_add(overhead_us, Ordering::Relaxed);
        self.launch_overhead_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_frame_latency(&self, latency_us: u64) {
        self.peak_frame_us.fetch_max(latency_us, Ordering::Relaxed);
    }

    fn avg(total: &AtomicU64, count: &AtomicU64) -> u64 {
        let c = count.load(Ordering::Relaxed);
        if c > 0 {
            total.load(Ordering::Relaxed) / c
        } else {
            0
        }
    }

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

const BUCKET_ALIGNMENT: usize = 2 * 1024 * 1024;
const MAX_PER_BUCKET: usize = 32;

struct BucketedPool {
    buckets: HashMap<usize, Vec<CudaSlice<u8>>>,
}

impl BucketedPool {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    fn take(&mut self, bucket_size: usize) -> Option<CudaSlice<u8>> {
        let stack = self.buckets.get_mut(&bucket_size)?;
        let buf = stack.pop()?;
        if stack.is_empty() {
            self.buckets.remove(&bucket_size);
        }
        Some(buf)
    }

    fn put(&mut self, buf: CudaSlice<u8>) -> Option<CudaSlice<u8>> {
        let bucket_size = buf.len();
        let stack = self.buckets.entry(bucket_size).or_default();
        if stack.len() >= MAX_PER_BUCKET {
            return Some(buf);
        }
        stack.push(buf);
        None
    }

    fn total_buffers(&self) -> usize {
        self.buckets.values().map(|s| s.len()).sum()
    }

    fn total_bytes(&self) -> usize {
        self.buckets
            .iter()
            .map(|(size, stack)| size * stack.len())
            .sum()
    }

    fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

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

#[inline]
fn bucket_for(size: usize) -> usize {
    if size == 0 {
        return BUCKET_ALIGNMENT.min(4096);
    }
    if size < BUCKET_ALIGNMENT {
        size.max(4096).next_power_of_two()
    } else {
        (size + BUCKET_ALIGNMENT - 1) & !(BUCKET_ALIGNMENT - 1)
    }
}

fn check_vram_limit(
    strict: bool,
    limit: Option<usize>,
    current: usize,
    requested: usize,
    reserve: usize,
) -> Result<bool> {
    let Some(limit_bytes) = limit.filter(|limit| *limit > 0) else {
        return Ok(false);
    };

    let would_be = current.saturating_add(reserve);
    if would_be <= limit_bytes {
        return Ok(false);
    }

    if strict {
        return Err(EngineError::VramLimitExceeded {
            limit_bytes,
            current_bytes: current,
            requested_bytes: requested,
            would_be_bytes: would_be,
        });
    }

    Ok(true)
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

impl Default for QueueDepthTracker {
    fn default() -> Self {
        Self::new()
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
    use crate::error::EngineError;

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

    #[test]
    fn vram_limit_warn_only_mode_preserves_success() {
        let ok = check_vram_limit(false, Some(1_024), 900, 200, 256)
            .expect("warn-only mode should not fail");
        assert!(ok, "helper should indicate would-exceed for caller warning");
    }

    #[test]
    fn vram_limit_strict_mode_fails_when_exceeded() {
        let err =
            check_vram_limit(true, Some(1_024), 900, 200, 256).expect_err("strict mode must fail");
        match err {
            EngineError::VramLimitExceeded {
                limit_bytes,
                current_bytes,
                requested_bytes,
                would_be_bytes,
            } => {
                assert_eq!(limit_bytes, 1_024);
                assert_eq!(current_bytes, 900);
                assert_eq!(requested_bytes, 200);
                assert_eq!(would_be_bytes, 1_156);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn vram_limit_strict_mode_allows_exact_boundary() {
        let ok =
            check_vram_limit(true, Some(1_024), 768, 128, 256).expect("equal-to-limit is allowed");
        assert!(!ok, "no warning when boundary is exactly met");
    }
}
