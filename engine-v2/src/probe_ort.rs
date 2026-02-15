//! ORT API exploration stub — used during development to verify IO Binding types.
//!
//! This module is intentionally minimal and will be removed before release.

pub fn probe(_binding: &mut ort::io_binding::IoBinding, _mem: &ort::memory::MemoryInfo) {
    let _api = ort::api();
}
