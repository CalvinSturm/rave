//! Debug-only host allocation tracker.
//!
//! Activated via `--features debug-alloc`.  Wraps the global allocator to
//! count host-side heap allocations during steady-state processing.
//!
//! # Usage
//!
//! ```rust,ignore
//! #[cfg(feature = "debug-alloc")]
//! {
//!     crate::debug_alloc::reset();
//!     crate::debug_alloc::enable();
//!     // ... inference stage runs one frame ...
//!     crate::debug_alloc::disable();
//!     let count = crate::debug_alloc::count();
//!     assert_eq!(count, 0, "host allocations during inference: {count}");
//! }
//! ```
//!
//! Does NOT affect release builds (feature-gated, zero-cost when disabled).

#[cfg(feature = "debug-alloc")]
pub use inner::*;

#[cfg(feature = "debug-alloc")]
mod inner {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static TRACKING: AtomicBool = AtomicBool::new(false);
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

    /// Global allocator wrapper that counts host allocations when enabled.
    pub struct TrackingAllocator;

    // SAFETY: delegates to `System` allocator for all actual work.
    // The atomic counter adds negligible overhead (single relaxed fetch_add).
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if TRACKING.load(Ordering::Relaxed) {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            // SAFETY: delegates to System which upholds GlobalAlloc contract.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: ptr was allocated by System.alloc with the same layout.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    /// Start counting host allocations.
    pub fn enable() {
        TRACKING.store(true, Ordering::Release);
    }

    /// Stop counting host allocations.
    pub fn disable() {
        TRACKING.store(false, Ordering::Release);
    }

    /// Reset the allocation counter to zero.
    pub fn reset() {
        ALLOC_COUNT.store(0, Ordering::Release);
    }

    /// Read the current allocation count.
    pub fn count() -> usize {
        ALLOC_COUNT.load(Ordering::Acquire)
    }
}

// Stub functions when feature is disabled — optimized away entirely.
#[cfg(not(feature = "debug-alloc"))]
pub fn enable() {}
#[cfg(not(feature = "debug-alloc"))]
pub fn disable() {}
#[cfg(not(feature = "debug-alloc"))]
pub fn reset() {}
#[cfg(not(feature = "debug-alloc"))]
pub fn count() -> usize {
    0
}
