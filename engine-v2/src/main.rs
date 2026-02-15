//! VideoForge v2.0 — CLI entrypoint.
//!
//! Parses user flags, validates hardware, builds the runtime, and runs
//! the GPU-resident upscale pipeline.
//!
//! ```bash
//! videoforge -i input.mp4 -o output.mp4 -m model.onnx
//! videoforge -i input.265 -o output.265 -m model.onnx --codec hevc --width 1920 --height 1080
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use videoforge_engine::backends::tensorrt::TensorRtBackend;
use videoforge_engine::core::backend::UpscaleBackend;
use videoforge_engine::codecs::nvdec::NvDecoder;
use videoforge_engine::codecs::nvenc::{NvEncConfig, NvEncoder};
use videoforge_engine::codecs::sys::cudaVideoCodec;
use videoforge_engine::core::context::GpuContext;
use videoforge_engine::core::kernels::{ModelPrecision, PreprocessKernels};
use videoforge_engine::engine::pipeline::{PipelineConfig, UpscalePipeline};
use videoforge_engine::error::Result;
use videoforge_engine::io::file_sink::FileBitstreamSink;
use videoforge_engine::io::file_source::FileBitstreamSource;
use videoforge_engine::io::ffmpeg_demuxer::FfmpegDemuxer;
use videoforge_engine::io::ffmpeg_muxer::FfmpegMuxer;
use videoforge_engine::io::probe_container;

// ─── CLI argument definition ─────────────────────────────────────────────────

/// VideoForge v2.0 — GPU-native video super-resolution engine.
///
/// Upscales video using NVDEC hardware decode, TensorRT/ONNX inference,
/// and NVENC hardware encode.  All processing is GPU-resident (zero host copies).
#[derive(Parser, Debug)]
#[command(name = "videoforge", version, about)]
struct Cli {
    /// Input video file (MP4/MKV/MOV containers or raw H.264/HEVC Annex B).
    ///
    /// Container files are auto-detected by extension and probed for
    /// codec/resolution/framerate.  Raw bitstreams (.264/.265/.hevc)
    /// require --codec, --width, and --height.
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Output video file (MP4/MKV/MOV containers or raw HEVC Annex B).
    ///
    /// Container format is auto-detected from the extension.
    /// Raw bitstream output uses .265/.hevc extensions.
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// ONNX model path for super-resolution.
    #[arg(short = 'm', long = "model")]
    model: PathBuf,

    /// Model precision: fp32 or fp16.
    #[arg(short = 'p', long = "precision", default_value = "fp32")]
    precision: String,

    /// CUDA device ordinal (0-indexed).
    #[arg(short = 'd', long = "device", default_value_t = 0)]
    device: usize,

    /// VRAM limit in MiB (0 = unlimited).
    #[arg(long = "vram-limit", default_value_t = 0)]
    vram_limit: usize,

    /// Channel capacity: decode → preprocess.
    #[arg(long = "decode-cap", default_value_t = 4)]
    decode_cap: usize,

    /// Channel capacity: preprocess → inference.
    #[arg(long = "preprocess-cap", default_value_t = 2)]
    preprocess_cap: usize,

    /// Channel capacity: inference → encode.
    #[arg(long = "upscale-cap", default_value_t = 4)]
    upscale_cap: usize,

    /// Output encoder bitrate in kbps (0 = CQP mode).
    #[arg(long = "bitrate", default_value_t = 0)]
    bitrate: u32,

    /// Input video codec: h264 or hevc (auto-detected for containers).
    #[arg(long = "codec")]
    codec: Option<String>,

    /// Output framerate numerator (auto-detected for containers).
    #[arg(long = "fps-num")]
    fps_num: Option<u32>,

    /// Output framerate denominator (auto-detected for containers).
    #[arg(long = "fps-den")]
    fps_den: Option<u32>,

    /// Input resolution width (auto-detected for containers).
    #[arg(long = "width")]
    width: Option<u32>,

    /// Input resolution height (auto-detected for containers).
    #[arg(long = "height")]
    height: Option<u32>,
}

// ─── Container detection ─────────────────────────────────────────────────────

/// Known container extensions that FFmpeg should handle.
const CONTAINER_EXTENSIONS: &[&str] = &["mp4", "mkv", "mov", "avi", "webm", "ts", "flv"];

/// Check if a path has a container extension.
fn is_container(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| CONTAINER_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    // Initialize tracing (structured logging).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    tracing::info!(
        input = %cli.input.display(),
        output = %cli.output.display(),
        model = %cli.model.display(),
        precision = %cli.precision,
        device = cli.device,
        "VideoForge v2.0 starting"
    );

    // Build and run the tokio runtime.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime");

    let result = rt.block_on(run_pipeline(cli));

    match result {
        Ok(()) => {
            tracing::info!("Pipeline completed successfully");
            std::process::exit(0);
        }
        Err(e) => {
            tracing::error!(error = %e, code = e.error_code(), "Pipeline failed");
            std::process::exit(e.error_code() as i32);
        }
    }
}

// ─── Pipeline orchestrator ───────────────────────────────────────────────────

async fn run_pipeline(cli: Cli) -> Result<()> {
    let wall_start = Instant::now();

    // ── 1. Parse precision ──
    let precision = match cli.precision.to_lowercase().as_str() {
        "fp32" | "f32" | "float32" => ModelPrecision::F32,
        "fp16" | "f16" | "float16" | "half" => ModelPrecision::F16,
        other => {
            return Err(videoforge_engine::error::EngineError::Pipeline(format!(
                "Unknown precision '{}'. Use fp32 or fp16.",
                other
            )));
        }
    };

    // ── 2. Detect input type and resolve codec/width/height/fps ──
    let input_is_container = is_container(&cli.input);
    let output_is_container = is_container(&cli.output);

    let (codec, input_width, input_height, fps_num, fps_den) = if input_is_container {
        // Probe the container for metadata.
        tracing::info!(path = %cli.input.display(), "Probing container for metadata");
        let meta = probe_container(&cli.input)?;

        // CLI overrides take precedence over probed values.
        let codec = if let Some(ref c) = cli.codec {
            parse_codec(c)?
        } else {
            meta.codec
        };
        let w = cli.width.unwrap_or(meta.width);
        let h = cli.height.unwrap_or(meta.height);
        let fnum = cli.fps_num.unwrap_or(meta.fps_num);
        let fden = cli.fps_den.unwrap_or(meta.fps_den);

        (codec, w, h, fnum, fden)
    } else {
        // Raw bitstream — use CLI args with defaults.
        let codec = parse_codec(cli.codec.as_deref().unwrap_or("hevc"))?;
        let w = cli.width.unwrap_or(1920);
        let h = cli.height.unwrap_or(1080);
        let fnum = cli.fps_num.unwrap_or(30);
        let fden = cli.fps_den.unwrap_or(1);

        (codec, w, h, fnum, fden)
    };

    tracing::info!(
        ?codec,
        width = input_width,
        height = input_height,
        fps = format!("{fps_num}/{fps_den}"),
        input_container = input_is_container,
        output_container = output_is_container,
        "Input parameters resolved"
    );

    // ── 3. Initialize GPU context ──
    tracing::info!(device = cli.device, "Initializing GPU context");
    let ctx = GpuContext::new(cli.device)?;

    if cli.vram_limit > 0 {
        let limit_bytes = cli.vram_limit * 1024 * 1024;
        ctx.set_vram_limit(limit_bytes);
    }

    // ── 4. Compile preprocessing kernels (NVRTC) ──
    tracing::info!("Compiling CUDA preprocessing kernels via NVRTC");
    let kernels = PreprocessKernels::compile(ctx.device())?;
    let kernels = Arc::new(kernels);

    // ── 5. Create TensorRT backend ──
    tracing::info!(model = %cli.model.display(), "Loading ONNX model");
    let backend = TensorRtBackend::new(
        cli.model.clone(),
        ctx.clone(),
        cli.device as i32,
        cli.upscale_cap + 2, // ring_size ≥ downstream_capacity + 2
        cli.upscale_cap,
    );
    let backend = Arc::new(backend);

    // Initialize backend (creates ORT session, builds TensorRT engine).
    tracing::info!("Initializing TensorRT backend (may take a while on first run)");
    backend.initialize().await?;

    // Query model metadata to compute output resolution.
    let meta = backend.metadata()?;
    let scale = meta.scale;
    let out_width = input_width * scale;
    let out_height = input_height * scale;

    tracing::info!(
        model_name = %meta.name,
        scale = scale,
        input_w = input_width,
        input_h = input_height,
        output_w = out_width,
        output_h = out_height,
        "Model metadata loaded"
    );

    // ── 6. Compute NV12 pitch for encoder ──
    // NVENC requires pitch aligned to 256 bytes.
    let nv12_pitch = ((out_width as usize + 255) / 256) * 256;

    // ── 7. Create decoder (NVDEC) with appropriate source ──
    tracing::info!(path = %cli.input.display(), ?codec, "Creating NVDEC decoder");
    let source: Box<dyn videoforge_engine::codecs::nvdec::BitstreamSource> = if input_is_container {
        Box::new(FfmpegDemuxer::new(&cli.input, codec)?)
    } else {
        Box::new(FileBitstreamSource::new(cli.input)?)
    };
    let decoder = NvDecoder::new(ctx.clone(), source, codec)?;

    // ── 8. Create encoder (NVENC) with appropriate sink ──
    tracing::info!(
        path = %cli.output.display(),
        width = out_width,
        height = out_height,
        "Creating NVENC encoder"
    );
    let sink: Box<dyn videoforge_engine::codecs::nvenc::BitstreamSink> = if output_is_container {
        Box::new(FfmpegMuxer::new(
            &cli.output,
            out_width,
            out_height,
            fps_num,
            fps_den,
        )?)
    } else {
        Box::new(FileBitstreamSink::new(cli.output)?)
    };

    let enc_config = NvEncConfig {
        width: out_width,
        height: out_height,
        fps_num,
        fps_den,
        bitrate: cli.bitrate * 1000, // kbps → bps
        max_bitrate: 0,
        gop_length: 250,
        b_frames: 0,
        nv12_pitch: nv12_pitch as u32,
    };

    // Get the raw CUDA context handle for NVENC.
    // SAFETY: cudarc's CudaDevice wraps a CUcontext.  We extract the raw
    // handle using the same transmute approach as get_raw_stream().
    let cuda_ctx_raw: *mut std::ffi::c_void = unsafe {
        let device = ctx.device();
        // CudaDevice stores the CUcontext as its first field.
        let ptr =
            device.as_ref() as *const cudarc::driver::CudaDevice as *const *mut std::ffi::c_void;
        *ptr
    };

    let encoder = NvEncoder::new(cuda_ctx_raw, sink, enc_config)?;

    // ── 9. Build pipeline ──
    let pipeline_config = PipelineConfig {
        decoded_capacity: cli.decode_cap,
        preprocessed_capacity: cli.preprocess_cap,
        upscaled_capacity: cli.upscale_cap,
        encoder_nv12_pitch: nv12_pitch,
        model_precision: precision,
        enable_profiler: true,
    };

    let pipeline = UpscalePipeline::new(ctx.clone(), kernels, pipeline_config);

    tracing::info!("Pipeline assembled — starting upscale run");

    // ── 10. Run pipeline ──
    pipeline.run(decoder, backend.clone(), encoder).await?;

    // ── 11. Shutdown and report ──
    let elapsed = wall_start.elapsed();

    // Report pool and VRAM stats.
    ctx.report_pool_stats();

    let (vram_current, vram_peak) = ctx.vram_usage();
    tracing::info!(
        elapsed_s = format!("{:.2}", elapsed.as_secs_f64()),
        vram_current_mb = vram_current / (1024 * 1024),
        vram_peak_mb = vram_peak / (1024 * 1024),
        "Engine shutdown complete"
    );

    // Shutdown backend (release ORT session, TRT engine cache).
    backend.shutdown().await?;

    Ok(())
}

/// Parse a codec string into a `cudaVideoCodec`.
fn parse_codec(s: &str) -> Result<cudaVideoCodec> {
    match s.to_lowercase().as_str() {
        "hevc" | "h265" | "265" => Ok(cudaVideoCodec::HEVC),
        "h264" | "264" | "avc" => Ok(cudaVideoCodec::H264),
        other => Err(videoforge_engine::error::EngineError::Pipeline(format!(
            "Unknown codec '{}'. Use hevc or h264.",
            other
        ))),
    }
}
