//! Minimal CUDA driver FFI helpers used by kernel timing and stream sync.

use rave_core::error::{EngineError, Result};
pub use rave_core::ffi_types::{CUDA_SUCCESS, CUevent, CUresult, CUstream};
#[cfg(target_os = "linux")]
use std::ffi::{CStr, CString, c_char, c_void};
use std::os::raw::c_uint;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

#[cfg(not(target_os = "linux"))]
unsafe extern "C" {
    fn cuEventCreate(phEvent: *mut CUevent, Flags: c_uint) -> CUresult;
    fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
    fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;
    fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: c_uint) -> CUresult;
    fn cuEventElapsedTime(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult;
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "linux")]
const RTLD_NOW: i32 = 2;
#[cfg(target_os = "linux")]
const RTLD_GLOBAL: i32 = 0x100;

#[cfg(target_os = "linux")]
struct CudaDriverApi {
    cu_event_create: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    cu_event_record: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    cu_event_destroy_v2: unsafe extern "C" fn(CUevent) -> CUresult,
    cu_stream_wait_event: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUresult,
    cu_event_elapsed_time: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,
}

#[cfg(target_os = "linux")]
static CUDA_DRIVER_API: OnceLock<std::result::Result<CudaDriverApi, String>> = OnceLock::new();

#[cfg(target_os = "linux")]
fn load_cuda_symbol<T>(handle: *mut c_void, name: &'static str) -> std::result::Result<T, String> {
    let cname = CString::new(name).map_err(|_| format!("invalid CUDA symbol name: {name}"))?;
    // SAFETY: handle is a valid dlopen handle and cname is a valid C symbol name.
    let ptr = unsafe { dlsym(handle, cname.as_ptr()) };
    if ptr.is_null() {
        // SAFETY: dlerror returns thread-local C string or null.
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
        // SAFETY: ptr points to a function with signature T.
        Ok(unsafe { std::mem::transmute_copy(&ptr) })
    }
}

#[cfg(target_os = "linux")]
fn init_cuda_driver_api() -> std::result::Result<CudaDriverApi, String> {
    let mut handle = std::ptr::null_mut();
    let mut last_err = "unknown dlopen error".to_string();
    for candidate in ["libcuda.so.1", "libcuda.so"] {
        let soname =
            CString::new(candidate).map_err(|_| format!("invalid CUDA soname: {candidate}"))?;
        // SAFETY: static soname and valid dlopen flags.
        handle = unsafe { dlopen(soname.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if !handle.is_null() {
            break;
        }
        // SAFETY: dlerror returns thread-local C string or null.
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

    Ok(CudaDriverApi {
        cu_event_create: load_cuda_symbol(handle, "cuEventCreate")?,
        cu_event_record: load_cuda_symbol(handle, "cuEventRecord")?,
        cu_event_destroy_v2: load_cuda_symbol(handle, "cuEventDestroy_v2")?,
        cu_stream_wait_event: load_cuda_symbol(handle, "cuStreamWaitEvent")?,
        cu_event_elapsed_time: load_cuda_symbol(handle, "cuEventElapsedTime")?,
    })
}

#[cfg(target_os = "linux")]
fn cuda_driver_api() -> Result<&'static CudaDriverApi> {
    let api = CUDA_DRIVER_API.get_or_init(init_cuda_driver_api);
    api.as_ref().map_err(|err| {
        EngineError::Decode(format!(
            "failed to load CUDA driver API: {err}. \
Ensure NVIDIA driver libraries are installed and visible via LD_LIBRARY_PATH \
(on WSL, prepend /usr/lib/wsl/lib)."
        ))
    })
}

/// Call `cuEventCreate`.
///
/// # Safety
/// `ph_event` must be a valid, writable pointer to CUDA event storage.
pub unsafe fn cu_event_create(ph_event: *mut CUevent, flags: c_uint) -> Result<CUresult> {
    #[cfg(target_os = "linux")]
    {
        let api = cuda_driver_api()?;
        // SAFETY: function pointer was resolved from CUDA driver with matching signature.
        Ok(unsafe { (api.cu_event_create)(ph_event, flags) })
    }
    #[cfg(not(target_os = "linux"))]
    {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuEventCreate(ph_event, flags) })
    }
}

/// Call `cuEventRecord`.
///
/// # Safety
/// `event` and `stream` must be valid CUDA driver handles from the same device/context.
pub unsafe fn cu_event_record(event: CUevent, stream: CUstream) -> Result<CUresult> {
    #[cfg(target_os = "linux")]
    {
        let api = cuda_driver_api()?;
        // SAFETY: function pointer was resolved from CUDA driver with matching signature.
        Ok(unsafe { (api.cu_event_record)(event, stream) })
    }
    #[cfg(not(target_os = "linux"))]
    {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuEventRecord(event, stream) })
    }
}

/// Call `cuEventDestroy_v2`.
///
/// # Safety
/// `event` must be a valid CUDA event handle (or null if tolerated by the driver).
pub unsafe fn cu_event_destroy_v2(event: CUevent) -> Result<CUresult> {
    #[cfg(target_os = "linux")]
    {
        let api = cuda_driver_api()?;
        // SAFETY: function pointer was resolved from CUDA driver with matching signature.
        Ok(unsafe { (api.cu_event_destroy_v2)(event) })
    }
    #[cfg(not(target_os = "linux"))]
    {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuEventDestroy_v2(event) })
    }
}

/// Call `cuStreamWaitEvent`.
///
/// # Safety
/// `stream` and `event` must be valid CUDA handles from the active context.
pub unsafe fn cu_stream_wait_event(
    stream: CUstream,
    event: CUevent,
    flags: c_uint,
) -> Result<CUresult> {
    #[cfg(target_os = "linux")]
    {
        let api = cuda_driver_api()?;
        // SAFETY: function pointer was resolved from CUDA driver with matching signature.
        Ok(unsafe { (api.cu_stream_wait_event)(stream, event, flags) })
    }
    #[cfg(not(target_os = "linux"))]
    {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuStreamWaitEvent(stream, event, flags) })
    }
}

/// Call `cuEventElapsedTime`.
///
/// # Safety
/// `ms` must be a valid writable pointer and `start`/`end` must be valid recorded CUDA events.
pub unsafe fn cu_event_elapsed_time(
    ms: *mut f32,
    start: CUevent,
    end: CUevent,
) -> Result<CUresult> {
    #[cfg(target_os = "linux")]
    {
        let api = cuda_driver_api()?;
        // SAFETY: function pointer was resolved from CUDA driver with matching signature.
        Ok(unsafe { (api.cu_event_elapsed_time)(ms, start, end) })
    }
    #[cfg(not(target_os = "linux"))]
    {
        // SAFETY: FFI call into CUDA driver API.
        Ok(unsafe { cuEventElapsedTime(ms, start, end) })
    }
}

#[inline]
pub fn check_cu(result: CUresult, context: &str) -> Result<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(EngineError::Decode(format!(
            "{context} failed with CUDA error code {result}"
        )))
    }
}
