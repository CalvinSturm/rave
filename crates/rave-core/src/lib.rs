#![doc = include_str!("../README.md")]

pub mod backend;
pub mod codec_traits;
pub mod context;
pub mod debug_alloc;
pub mod error;
pub mod ffi_types;
pub mod host_copy_audit;
pub mod types;

#[macro_export]
macro_rules! host_copy_violation {
    ($stage:expr, $($arg:tt)+) => {{
        #[cfg(feature = "audit-no-host-copies")]
        {
            $crate::host_copy_audit::record_violation($stage, format!($($arg)+));
        }
        #[cfg(not(feature = "audit-no-host-copies"))]
        {
            let _ = &$stage;
            let _ = format_args!($($arg)+);
        }
    }};
}
