//! CUDA stream utility helpers shared across crates.

use cudarc::driver::CudaStream;
use rave_core::error::Result;
use rave_core::ffi_types::{CUevent, CUstream};

use crate::sys;

#[inline]
pub fn get_raw_stream(stream: &CudaStream) -> CUstream {
    stream.stream as CUstream
}

#[inline]
// `event` is an opaque CUDA driver handle passed through to the driver API.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn wait_for_event(target_stream: &CudaStream, event: CUevent) -> Result<()> {
    let raw_stream = get_raw_stream(target_stream);
    unsafe {
        sys::check_cu(
            sys::cuStreamWaitEvent(raw_stream, event, 0),
            "cuStreamWaitEvent",
        )?;
    }
    Ok(())
}
