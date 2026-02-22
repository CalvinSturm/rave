//! Feature-gated host-copy audit helpers.
//!
//! This module is intentionally tiny: when `audit-no-host-copies` is disabled,
//! all helpers compile down to no-ops.

use crate::error::{EngineError, Result};
use crate::types::{GpuTexture, PixelFormat};
use cudarc::driver::DeviceSlice;

#[cfg(feature = "audit-no-host-copies")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "audit-no-host-copies")]
static STRICT_MODE: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "audit-no-host-copies")]
static WARNED_ONCE: AtomicBool = AtomicBool::new(false);

/// Restores the previous strict mode value when dropped.
pub struct StrictModeGuard {
    #[cfg(feature = "audit-no-host-copies")]
    previous: bool,
}

impl Drop for StrictModeGuard {
    fn drop(&mut self) {
        #[cfg(feature = "audit-no-host-copies")]
        {
            STRICT_MODE.store(self.previous, Ordering::Relaxed);
        }
    }
}

/// Enable or disable strict no-host-copies mode for the current run scope.
#[must_use]
pub fn push_strict_mode(enabled: bool) -> StrictModeGuard {
    #[cfg(feature = "audit-no-host-copies")]
    {
        let previous = STRICT_MODE.swap(enabled, Ordering::Relaxed);
        return StrictModeGuard { previous };
    }

    #[cfg(not(feature = "audit-no-host-copies"))]
    {
        let _ = enabled;
        StrictModeGuard {}
    }
}

/// Whether strict no-host-copies mode is currently enabled.
pub fn is_strict_mode() -> bool {
    #[cfg(feature = "audit-no-host-copies")]
    {
        return STRICT_MODE.load(Ordering::Relaxed);
    }

    #[cfg(not(feature = "audit-no-host-copies"))]
    {
        false
    }
}

/// Record a suspicious host-copy path.
///
/// In strict mode this triggers a debug assertion. In non-strict mode this logs
/// a single warning for the process lifetime.
pub fn record_violation(stage: &str, detail: String) {
    #[cfg(feature = "audit-no-host-copies")]
    {
        if is_strict_mode() {
            debug_assert!(
                false,
                "no-host-copies strict violation at stage `{stage}`: {detail}"
            );
            return;
        }

        if !WARNED_ONCE.swap(true, Ordering::Relaxed) {
            tracing::warn!(stage, detail = %detail, "no-host-copies audit warning");
        }
    }

    #[cfg(not(feature = "audit-no-host-copies"))]
    {
        let _ = stage;
        let _ = detail;
    }
}

fn audit_enabled() -> bool {
    cfg!(feature = "audit-no-host-copies")
}

fn violation_detail(stage: &str, name: &str, ptr: u64, bytes: usize, reason: &str) -> String {
    format!(
        "no-host-copies sentinel failed: stage={stage} name={name} ptr=0x{ptr:016x} bytes={bytes} reason={reason}"
    )
}

fn handle_violation(stage: &'static str, detail: String, strict: bool) -> Result<()> {
    if strict {
        return Err(EngineError::InvariantViolation(detail));
    }

    record_violation(stage, detail);
    Ok(())
}

fn audit_device_ptr_impl(
    stage: &'static str,
    name: &'static str,
    ptr: u64,
    bytes: usize,
    enabled: bool,
    strict: bool,
) -> Result<()> {
    if !enabled {
        return Ok(());
    }

    if ptr == 0 {
        return handle_violation(
            stage,
            violation_detail(stage, name, ptr, bytes, "device pointer is null"),
            strict,
        );
    }

    if bytes == 0 {
        return handle_violation(
            stage,
            violation_detail(stage, name, ptr, bytes, "buffer size is zero"),
            strict,
        );
    }

    Ok(())
}

/// Stage boundary sentinel for raw device pointers.
///
/// This is a no-op when `audit-no-host-copies` is not enabled.
pub fn audit_device_ptr(
    stage: &'static str,
    name: &'static str,
    ptr: u64,
    bytes: usize,
    strict: bool,
) -> Result<()> {
    audit_device_ptr_impl(stage, name, ptr, bytes, audit_enabled(), strict)
}

/// Stage boundary sentinel for GPU textures.
///
/// This validates pointer presence and basic size/stride invariants used by
/// decode/preprocess/inference/encode handoffs.
///
/// This is a no-op when `audit-no-host-copies` is not enabled.
pub fn audit_device_texture(stage: &'static str, tex: &GpuTexture, strict: bool) -> Result<()> {
    if !audit_enabled() {
        return Ok(());
    }

    let ptr = tex.device_ptr();
    let expected_bytes = tex.byte_size();
    let alloc_bytes = tex.data.len();
    audit_device_ptr_impl(stage, "texture", ptr, expected_bytes, true, strict)?;

    if alloc_bytes < expected_bytes {
        return handle_violation(
            stage,
            violation_detail(
                stage,
                "texture",
                ptr,
                expected_bytes,
                &format!("device allocation too small (have={alloc_bytes}, need={expected_bytes})"),
            ),
            strict,
        );
    }

    let min_pitch = match tex.format {
        PixelFormat::Nv12 | PixelFormat::RgbInterleavedU8 => tex.width as usize,
        PixelFormat::RgbPlanarF32 => tex.width as usize * 4,
        PixelFormat::RgbPlanarF16 => tex.width as usize * 2,
    };
    if tex.pitch < min_pitch {
        return handle_violation(
            stage,
            violation_detail(
                stage,
                "texture",
                ptr,
                expected_bytes,
                &format!(
                    "pitch is smaller than minimum for format {:?} (pitch={}, min_pitch={min_pitch})",
                    tex.format, tex.pitch
                ),
            ),
            strict,
        );
    }

    let alignment = tex.format.element_bytes() as u64;
    if alignment > 1 && !ptr.is_multiple_of(alignment) {
        return handle_violation(
            stage,
            violation_detail(
                stage,
                "texture",
                ptr,
                expected_bytes,
                &format!("pointer is not aligned to element size (alignment={alignment})"),
            ),
            strict,
        );
    }

    Ok(())
}

#[cfg(all(test, feature = "audit-no-host-copies"))]
mod tests {
    use super::{is_strict_mode, push_strict_mode};

    #[test]
    fn strict_mode_guard_restores_previous_value() {
        let original = is_strict_mode();
        {
            let _guard = push_strict_mode(true);
            assert!(is_strict_mode());
        }
        assert_eq!(is_strict_mode(), original);
    }
}

#[cfg(test)]
mod sentinel_tests {
    use super::audit_device_ptr_impl;

    #[test]
    fn device_ptr_audit_disabled_is_noop() {
        audit_device_ptr_impl("decode->preprocess", "decoded.texture", 0, 0, false, true)
            .expect("audit disabled should always be ok");
    }

    #[test]
    fn device_ptr_audit_enabled_non_strict_warn_path_is_non_fatal() {
        audit_device_ptr_impl("decode->preprocess", "decoded.texture", 0, 128, true, false)
            .expect("non-strict audit should warn without failing");
    }

    #[test]
    fn device_ptr_audit_enabled_strict_rejects_null_ptr() {
        let err =
            audit_device_ptr_impl("decode->preprocess", "decoded.texture", 0, 128, true, true)
                .expect_err("strict audit must reject null ptr");
        let msg = err.to_string();
        assert!(msg.contains("decode->preprocess"));
        assert!(msg.contains("ptr=0x0000000000000000"));
    }

    #[test]
    fn device_ptr_audit_enabled_strict_accepts_non_null_ptr() {
        audit_device_ptr_impl(
            "inference->encode",
            "upscaled.texture",
            0x1000,
            4096,
            true,
            true,
        )
        .expect("strict audit should accept non-zero pointers");
    }
}
