//! CLI/runtime bridge helpers.
//!
//! This module keeps low-level composition inside `rave-pipeline` so callers
//! can depend on pipeline contracts without directly importing codec/container/
//! backend crates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rave_core::backend::UpscaleBackend;
use rave_core::codec_traits::{BitstreamSink, BitstreamSource};
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec;
use rave_cuda::kernels::{ModelPrecision, PreprocessKernels};
use rave_ffmpeg::ffmpeg_demuxer::FfmpegDemuxer;
use rave_ffmpeg::ffmpeg_muxer::FfmpegMuxer;
use rave_ffmpeg::file_sink::FileBitstreamSink;
use rave_ffmpeg::file_source::FileBitstreamSource;
use rave_ffmpeg::probe_container;
use rave_nvcodec::nvdec::NvDecoder;
use rave_nvcodec::nvenc::{NvEncConfig, NvEncoder};
use rave_tensorrt::tensorrt::TensorRtBackend;

pub type Decoder = NvDecoder;
pub type Encoder = NvEncoder;

/// Known container extensions.
pub const CONTAINER_EXTENSIONS: &[&str] = &["mp4", "mkv", "mov", "avi", "webm", "ts", "flv"];

#[derive(Debug, Clone, Copy)]
pub struct ResolvedInput {
    pub codec: cudaVideoCodec,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub input_is_container: bool,
}

pub struct RuntimeSetup {
    pub ctx: Arc<GpuContext>,
    pub kernels: Arc<PreprocessKernels>,
    pub backend: Arc<TensorRtBackend>,
    pub precision: ModelPrecision,
    pub decoded_capacity: usize,
    pub preprocessed_capacity: usize,
    pub upscaled_capacity: usize,
    pub input: ResolvedInput,
    pub out_width: u32,
    pub out_height: u32,
    pub nv12_pitch: usize,
}

pub struct RuntimeRequest {
    pub model_path: PathBuf,
    pub precision: String,
    pub device: usize,
    pub vram_limit_mib: Option<usize>,
    pub strict_vram_limit: bool,
    pub decode_cap: Option<usize>,
    pub preprocess_cap: Option<usize>,
    pub upscale_cap: Option<usize>,
    pub resolved_input: ResolvedInput,
}

pub fn is_container(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| CONTAINER_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn parse_precision(s: &str) -> Result<ModelPrecision> {
    match s.to_ascii_lowercase().as_str() {
        "fp32" | "f32" | "float32" => Ok(ModelPrecision::F32),
        "fp16" | "f16" | "float16" | "half" => Ok(ModelPrecision::F16),
        other => Err(EngineError::Pipeline(format!(
            "Unknown precision '{other}'. Use fp32 or fp16."
        ))),
    }
}

pub fn parse_codec(s: &str) -> Result<cudaVideoCodec> {
    match s.to_ascii_lowercase().as_str() {
        "hevc" | "h265" | "265" => Ok(cudaVideoCodec::HEVC),
        "h264" | "264" | "avc" => Ok(cudaVideoCodec::H264),
        other => Err(EngineError::Pipeline(format!(
            "Unknown codec '{other}'. Use hevc or h264."
        ))),
    }
}

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

pub async fn prepare_runtime(request: &RuntimeRequest) -> Result<RuntimeSetup> {
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

pub fn create_decoder(setup: &RuntimeSetup, input_path: &Path) -> Result<NvDecoder> {
    let source: Box<dyn BitstreamSource> = if setup.input.input_is_container {
        Box::new(FfmpegDemuxer::new(input_path, setup.input.codec)?)
    } else {
        Box::new(FileBitstreamSource::new(input_path.to_path_buf())?)
    };
    NvDecoder::new(setup.ctx.clone(), source, setup.input.codec)
}

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
