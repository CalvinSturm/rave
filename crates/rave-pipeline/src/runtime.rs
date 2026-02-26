//! Compatibility re-exports for the concrete NVIDIA runtime helpers.
//!
//! Prefer importing these items from `rave_runtime_nvidia` directly.
//! This shim is temporary and is only available when the
//! `compat-runtime-nvidia` feature is enabled.

#[deprecated(
    since = "0.4.0",
    note = "Import from `rave_runtime_nvidia` instead (this shim will be removed in a future release)"
)]
pub use rave_runtime_nvidia::{
    CONTAINER_EXTENSIONS, Decoder, Encoder, ResolvedInput, RuntimeRequest, RuntimeSetup,
    create_context_and_kernels, create_decoder, create_nvenc_encoder, is_container, parse_codec,
    parse_precision, prepare_runtime, resolve_input,
};
