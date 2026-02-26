#![doc = include_str!("../README.md")]

//! Concrete NVIDIA runtime composition helpers.
//!
//! This crate owns the low-level composition of CUDA kernels, TensorRT,
//! FFmpeg container I/O, and NVDEC/NVENC so callers can depend on a concrete
//! runtime stack without coupling `rave-pipeline` to backend crates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rave_core::backend::UpscaleBackend;
use rave_core::codec_traits::{BitstreamSink, BitstreamSource};
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec;
use rave_core::host_copy_audit::require_host_copy_audit_if_strict;
use rave_cuda::kernels::{ModelPrecision, PreprocessKernels};
use rave_ffmpeg::ffmpeg_demuxer::FfmpegDemuxer;
use rave_ffmpeg::ffmpeg_muxer::FfmpegMuxer;
use rave_ffmpeg::file_sink::FileBitstreamSink;
use rave_ffmpeg::file_source::FileBitstreamSource;
use rave_ffmpeg::probe_container;
use rave_nvcodec::nvdec::NvDecoder;
use rave_nvcodec::nvenc::{NvEncConfig, NvEncoder};
use rave_tensorrt::tensorrt::TensorRtBackend;

/// Concrete decoder type used by the pipeline (NVDEC).
pub type Decoder = NvDecoder;
/// Concrete encoder type used by the pipeline (NVENC).
pub type Encoder = NvEncoder;

/// Known container extensions.
pub const CONTAINER_EXTENSIONS: &[&str] = &["mp4", "mkv", "mov", "avi", "webm", "ts", "flv"];

/// Resolved codec and geometry parameters for an input file.
///
/// Produced by [`resolve_input`] after probing a container or applying overrides.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedInput {
    /// Video codec in the input stream.
    pub codec: cudaVideoCodec,
    /// Coded frame width in pixels.
    pub width: u32,
    /// Coded frame height in pixels.
    pub height: u32,
    /// Framerate numerator.
    pub fps_num: u32,
    /// Framerate denominator.
    pub fps_den: u32,
    /// `true` if the input is a container (MP4/MKV/…); `false` for raw Annex B.
    pub input_is_container: bool,
}

/// Fully initialised GPU context and inference backend, ready to run the pipeline.
pub struct RuntimeSetup {
    /// GPU context with pooled VRAM allocator.
    pub ctx: Arc<GpuContext>,
    /// Compiled CUDA preprocess kernels.
    pub kernels: Arc<PreprocessKernels>,
    /// Initialised TensorRT inference backend.
    pub backend: Arc<TensorRtBackend>,
    /// Floating-point precision used by the preprocess/postprocess kernels.
    pub precision: ModelPrecision,
    /// Bounded channel capacity between decoder and preprocess stages.
    pub decoded_capacity: usize,
    /// Bounded channel capacity between preprocess and inference stages.
    pub preprocessed_capacity: usize,
    /// Bounded channel capacity between inference and encoder stages.
    pub upscaled_capacity: usize,
    /// Resolved input codec and geometry.
    pub input: ResolvedInput,
    /// Output frame width after upscaling in pixels.
    pub out_width: u32,
    /// Output frame height after upscaling in pixels.
    pub out_height: u32,
    /// NV12 row pitch aligned to 256 bytes, used by NVENC.
    pub nv12_pitch: usize,
}

/// Parameters passed to [`prepare_runtime`] by the CLI or a custom driver.
pub struct RuntimeRequest {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Precision string (`"fp32"` or `"fp16"`).
    pub precision: String,
    /// CUDA device ordinal.
    pub device: usize,
    /// Optional VRAM budget in MiB. `None` means unlimited.
    pub vram_limit_mib: Option<usize>,
    /// If `true`, the pool allocator will hard-fail on VRAM budget overrun.
    pub strict_vram_limit: bool,
    /// If `true`, the host-copy audit feature flag is required.
    pub strict_no_host_copies: bool,
    /// Override for the decode→preprocess channel capacity.
    pub decode_cap: Option<usize>,
    /// Override for the preprocess→inference channel capacity.
    pub preprocess_cap: Option<usize>,
    /// Override for the inference→encode channel capacity.
    pub upscale_cap: Option<usize>,
    /// Pre-resolved input codec and geometry from [`resolve_input`].
    pub resolved_input: ResolvedInput,
}

/// Return `true` if `path` has a known container file extension (MP4, MKV, etc.).
pub fn is_container(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| CONTAINER_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Parse a precision string (`"fp32"`, `"f32"`, `"fp16"`, `"f16"`, `"half"`) into a
/// [`ModelPrecision`]. Returns an error for unrecognised strings.
pub fn parse_precision(s: &str) -> Result<ModelPrecision> {
    match s.to_ascii_lowercase().as_str() {
        "fp32" | "f32" | "float32" => Ok(ModelPrecision::F32),
        "fp16" | "f16" | "float16" | "half" => Ok(ModelPrecision::F16),
        other => Err(EngineError::Pipeline(format!(
            "Unknown precision '{other}'. Use fp32 or fp16."
        ))),
    }
}

/// Parse a codec string (`"hevc"`, `"h264"`, etc.) into a [`cudaVideoCodec`].
/// Returns an error for unrecognised strings.
pub fn parse_codec(s: &str) -> Result<cudaVideoCodec> {
    match s.to_ascii_lowercase().as_str() {
        "hevc" | "h265" | "265" => Ok(cudaVideoCodec::HEVC),
        "h264" | "264" | "avc" => Ok(cudaVideoCodec::H264),
        other => Err(EngineError::Pipeline(format!(
            "Unknown codec '{other}'. Use hevc or h264."
        ))),
    }
}

/// Probe or construct a [`ResolvedInput`] for the given path.
///
/// For container files, calls FFmpeg probe and merges any CLI overrides.
/// For raw bitstream files, uses the provided overrides or defaults
/// (`HEVC`, 1920×1080, 30/1 fps).
#[allow(clippy::too_many_arguments)]
pub fn resolve_input(
    input: &Path,
    codec_override: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    fps_num: Option<u32>,
    fps_den: Option<u32>,
) -> Result<ResolvedInput> {
    let input_is_container = is_container(input);
    if input_is_container {
        let meta = probe_container(input)?;
        let codec = if let Some(c) = codec_override {
            parse_codec(c)?
        } else {
            meta.codec
        };
        Ok(ResolvedInput {
            codec,
            width: width.unwrap_or(meta.width),
            height: height.unwrap_or(meta.height),
            fps_num: fps_num.unwrap_or(meta.fps_num),
            fps_den: fps_den.unwrap_or(meta.fps_den),
            input_is_container,
        })
    } else {
        Ok(ResolvedInput {
            codec: parse_codec(codec_override.unwrap_or("hevc"))?,
            width: width.unwrap_or(1920),
            height: height.unwrap_or(1080),
            fps_num: fps_num.unwrap_or(30),
            fps_den: fps_den.unwrap_or(1),
            input_is_container,
        })
    }
}

/// Initialise GPU context, compile kernels, and warm up the TensorRT backend.
///
/// This is the primary entry point for setting up the pipeline. All GPU
/// resources are allocated here; no further allocation occurs during the run.
pub async fn prepare_runtime(request: &RuntimeRequest) -> Result<RuntimeSetup> {
    require_host_copy_audit_if_strict(request.strict_no_host_copies)?;

    let precision = parse_precision(&request.precision)?;

    let ctx = GpuContext::new(request.device)?;
    if request.strict_vram_limit {
        ctx.set_strict_vram_limit(true);
    }
    if let Some(vram_limit_mib) = request.vram_limit_mib
        && vram_limit_mib > 0
    {
        ctx.set_vram_limit(vram_limit_mib * 1024 * 1024);
    }

    let kernels = Arc::new(PreprocessKernels::compile(ctx.device())?);

    let decoded_capacity = request.decode_cap.unwrap_or(4);
    let preprocessed_capacity = request.preprocess_cap.unwrap_or(2);
    let upscaled_capacity = request.upscale_cap.unwrap_or(4);

    let backend = Arc::new(TensorRtBackend::new(
        request.model_path.clone(),
        ctx.clone(),
        request.device as i32,
        upscaled_capacity + 2,
        upscaled_capacity,
    ));
    backend.initialize().await?;

    let model_meta = backend.metadata()?;
    let out_width = request.resolved_input.width * model_meta.scale;
    let out_height = request.resolved_input.height * model_meta.scale;
    let nv12_pitch = (out_width as usize).div_ceil(256) * 256;

    Ok(RuntimeSetup {
        ctx,
        kernels,
        backend,
        precision,
        decoded_capacity,
        preprocessed_capacity,
        upscaled_capacity,
        input: request.resolved_input,
        out_width,
        out_height,
        nv12_pitch,
    })
}

/// Create a GPU context and compile preprocess kernels without initialising an inference backend.
///
/// Used by the `benchmark` and `devices` commands that need a context but not a full runtime.
pub fn create_context_and_kernels(
    device: usize,
    vram_limit_mib: Option<usize>,
    strict_vram_limit: bool,
) -> Result<(Arc<GpuContext>, Arc<PreprocessKernels>)> {
    let ctx = GpuContext::new(device)?;
    if strict_vram_limit {
        ctx.set_strict_vram_limit(true);
    }
    if let Some(vram_limit_mib) = vram_limit_mib
        && vram_limit_mib > 0
    {
        ctx.set_vram_limit(vram_limit_mib * 1024 * 1024);
    }
    let kernels = Arc::new(PreprocessKernels::compile(ctx.device())?);
    Ok((ctx, kernels))
}

/// Create an NVDEC decoder, automatically selecting a container demuxer or a
/// raw file source based on [`ResolvedInput::input_is_container`].
pub fn create_decoder(setup: &RuntimeSetup, input_path: &Path) -> Result<NvDecoder> {
    let source: Box<dyn BitstreamSource> = if setup.input.input_is_container {
        Box::new(FfmpegDemuxer::new(input_path, setup.input.codec)?)
    } else {
        Box::new(FileBitstreamSource::new(input_path.to_path_buf())?)
    };
    NvDecoder::new(setup.ctx.clone(), source, setup.input.codec)
}

/// Create an NVENC encoder, automatically selecting an FFmpeg muxer (for container
/// output paths) or a raw file sink (for `.265`/`.hevc` paths).
pub fn create_nvenc_encoder(
    setup: &RuntimeSetup,
    output_path: &Path,
    bitrate_kbps: u32,
    fps_num: u32,
    fps_den: u32,
) -> Result<NvEncoder> {
    let sink: Box<dyn BitstreamSink> = if is_container(output_path) {
        Box::new(FfmpegMuxer::new(
            output_path,
            setup.out_width,
            setup.out_height,
            fps_num,
            fps_den,
        )?)
    } else {
        Box::new(FileBitstreamSink::new(output_path.to_path_buf())?)
    };

    let enc_config = NvEncConfig {
        width: setup.out_width,
        height: setup.out_height,
        fps_num,
        fps_den,
        bitrate: bitrate_kbps * 1000,
        max_bitrate: 0,
        gop_length: 250,
        b_frames: 0,
        nv12_pitch: setup.nv12_pitch as u32,
    };

    let cuda_ctx_raw: *mut std::ffi::c_void =
        *setup.ctx.device().cu_primary_ctx() as *mut std::ffi::c_void;
    NvEncoder::new(cuda_ctx_raw, sink, enc_config)
}
