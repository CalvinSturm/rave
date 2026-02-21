//! Feature-gated host-copy audit helpers.
//!
//! This module is intentionally tiny: when `audit-no-host-copies` is disabled,
//! all helpers compile down to no-ops.

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
