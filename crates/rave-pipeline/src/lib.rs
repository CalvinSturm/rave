#![doc = include_str!("../README.md")]

pub mod inference;
pub mod pipeline;

pub use pipeline::{PipelineConfig, PipelineMetrics, UpscalePipeline};

#[cfg(test)]
mod tests {
    use super::PipelineConfig;

    #[test]
    fn strict_no_host_copies_defaults_to_off() {
        assert!(!PipelineConfig::default().strict_no_host_copies);
    }

    #[cfg(feature = "audit-no-host-copies")]
    #[test]
    fn audit_feature_wires_to_core_guard() {
        let baseline = rave_core::host_copy_audit::is_strict_mode();
        {
            let _guard = rave_core::host_copy_audit::push_strict_mode(true);
            assert!(rave_core::host_copy_audit::is_strict_mode());
        }
        assert_eq!(rave_core::host_copy_audit::is_strict_mode(), baseline);
    }
}
