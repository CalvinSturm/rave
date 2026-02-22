//! RAVE CLI entrypoint.
//!
//! ```bash
//! rave upscale --input input.mp4 --output output.mp4 --model model.onnx
//! rave benchmark --input input.mp4 --model model.onnx --json --json-out bench.json
//! rave benchmark --input input.mp4 --model model.onnx --progress jsonl
//! rave devices --json
//! rave probe
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{io::IsTerminal, sync::atomic::Ordering};

use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Deserialize;

use rave_core::backend::UpscaleBackend;
use rave_core::codec_traits::FrameEncoder;
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec;
use rave_core::types::FrameEnvelope;
use rave_pipeline::pipeline::{PipelineConfig, PipelineMetrics, UpscalePipeline};
use rave_pipeline::runtime::{
    Decoder as PipelineDecoder, Encoder as PipelineEncoder, ResolvedInput, RuntimeRequest,
    RuntimeSetup, create_context_and_kernels as pipeline_create_context_and_kernels,
    create_decoder as pipeline_create_decoder,
    create_nvenc_encoder as pipeline_create_nvenc_encoder,
    parse_precision as pipeline_parse_precision, prepare_runtime as pipeline_prepare_runtime,
    resolve_input as pipeline_resolve_input,
};
use rave_pipeline::{
    AuditLevel, BatchConfig as GraphBatchConfig, EnhanceConfig, GRAPH_SCHEMA_VERSION,
    PipelineReport, PrecisionPolicyConfig, ProfilePreset, RunContract, StageConfig, StageGraph,
    StageId,
};

#[cfg(target_os = "linux")]
use std::collections::HashSet;
#[cfg(target_os = "linux")]
use std::ffi::{CStr, c_char};
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
#[cfg(target_os = "linux")]
use std::time::UNIX_EPOCH;

#[derive(Parser, Debug)]
#[command(
    name = "rave",
    version,
    about = "GPU-native video engine",
    arg_required_else_help = true,
    after_help = "Examples:\n  rave probe --json\n  rave upscale --input in.mp4 --output out.mp4 --model model.onnx\n  rave benchmark --input in.mp4 --model model.onnx --skip-encode --json\n  rave benchmark --input in.mp4 --model model.onnx --skip-encode --json-out /tmp/bench.json\n  rave benchmark --input in.mp4 --model model.onnx --skip-encode --progress jsonl"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run full decode → infer → encode upscale pipeline.
    Upscale(UpscaleArgs),
    /// Run benchmark and emit summary.
    Benchmark(BenchmarkArgs),
    /// Run strict validation harness (graph + profile + contract checks).
    Validate(ValidateArgs),
    /// List visible CUDA devices and memory capacity.
    Devices(DevicesArgs),
    /// Probe CUDA context initialization and print basic status.
    Probe(ProbeArgs),
}

#[derive(Args, Debug, Clone)]
struct SharedVideoArgs {
    /// Input video file (container or raw .264/.265 bitstream).
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// ONNX model path.
    #[arg(short = 'm', long = "model")]
    model: PathBuf,

    /// Runtime profile preset.
    #[arg(long = "profile", value_enum, default_value_t = ProfileArg::Dev)]
    profile: ProfileArg,

    /// Optional JSON stage graph path.
    #[arg(long = "graph")]
    graph: Option<PathBuf>,

    /// Model precision: fp32 or fp16.
    #[arg(short = 'p', long = "precision")]
    precision: Option<String>,

    /// CUDA device ordinal.
    #[arg(short = 'd', long = "device")]
    device: Option<u32>,

    /// VRAM limit in MiB (0 = unlimited).
    #[arg(long = "vram-limit")]
    vram_limit: Option<usize>,

    /// Channel capacity: decode -> preprocess.
    #[arg(long = "decode-cap")]
    decode_cap: Option<usize>,

    /// Channel capacity: preprocess -> inference.
    #[arg(long = "preprocess-cap")]
    preprocess_cap: Option<usize>,

    /// Channel capacity: inference -> encode.
    #[arg(long = "upscale-cap")]
    upscale_cap: Option<usize>,

    /// Input codec override for raw bitstreams (h264/hevc).
    #[arg(long = "codec")]
    codec: Option<String>,

    /// Framerate numerator override.
    #[arg(long = "fps-num")]
    fps_num: Option<u32>,

    /// Framerate denominator override.
    #[arg(long = "fps-den")]
    fps_den: Option<u32>,

    /// Input width override.
    #[arg(long = "width")]
    width: Option<u32>,

    /// Input height override.
    #[arg(long = "height")]
    height: Option<u32>,
}

#[derive(Args, Debug, Clone)]
struct UpscaleArgs {
    #[command(flatten)]
    shared: SharedVideoArgs,

    /// Output video file (container or raw .265).
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Output encoder bitrate in kbps (0 = CQP).
    #[arg(long = "bitrate")]
    bitrate: Option<u32>,

    /// Parse/resolve inputs only, skip GPU execution.
    #[arg(long = "dry-run", default_value_t = false)]
    dry_run: bool,

    /// Emit structured JSON output to stdout.
    #[arg(long = "json", default_value_t = false)]
    json: bool,

    /// Progress output mode to stderr: auto (TTY only), off, human, jsonl.
    #[arg(long = "progress", value_enum, default_value_t = ProgressArg::Auto)]
    progress: ProgressArg,

    /// Emit progress as JSONL records to stderr.
    #[arg(long = "jsonl", default_value_t = false)]
    jsonl: bool,
}

#[derive(Args, Debug, Clone)]
struct BenchmarkArgs {
    #[command(flatten)]
    shared: SharedVideoArgs,

    /// Optional encoded benchmark output path.
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,

    /// Output encoder bitrate in kbps when encode is enabled.
    #[arg(long = "bitrate")]
    bitrate: Option<u32>,

    /// Force benchmark encode stage off.
    #[arg(long = "skip-encode", default_value_t = false)]
    skip_encode: bool,

    /// If NVENC initialization/run fails, rerun with encode skipped.
    #[arg(long = "allow-encode-skip", default_value_t = true)]
    allow_encode_skip: bool,

    /// Optional path to also write benchmark JSON.
    #[arg(long = "json-out")]
    json_out: Option<PathBuf>,

    /// Emit benchmark JSON to stdout (human-readable summary by default).
    #[arg(long = "json", default_value_t = false)]
    json: bool,

    /// Parse/resolve inputs only, emit synthetic benchmark summary.
    #[arg(long = "dry-run", default_value_t = false)]
    dry_run: bool,

    /// Progress output mode to stderr: auto (TTY only), off, human, jsonl.
    #[arg(long = "progress", value_enum, default_value_t = ProgressArg::Auto)]
    progress: ProgressArg,

    /// Emit progress as JSONL records to stderr.
    #[arg(long = "jsonl", default_value_t = false)]
    jsonl: bool,
}

#[derive(Args, Debug, Clone)]
struct ValidateArgs {
    /// Optional validate fixture JSON path (preconfigured scenario).
    #[arg(long = "fixture")]
    fixture: Option<PathBuf>,

    /// Input video file.
    #[arg(short = 'i', long = "input")]
    input: Option<PathBuf>,

    /// ONNX model path (used when `--graph` is omitted).
    #[arg(short = 'm', long = "model")]
    model: Option<PathBuf>,

    /// Optional output path used for validation runs.
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,

    /// Runtime profile preset.
    #[arg(long = "profile", value_enum, default_value_t = ProfileArg::ProductionStrict)]
    profile: ProfileArg,

    /// Optional JSON stage graph path.
    #[arg(long = "graph")]
    graph: Option<PathBuf>,

    /// CUDA device ordinal.
    #[arg(short = 'd', long = "device")]
    device: Option<u32>,

    /// Determinism run count for strict comparison.
    #[arg(long = "determinism-runs", default_value_t = 2)]
    determinism_runs: u32,

    /// Number of per-frame canonical hashes to collect.
    #[arg(long = "determinism-samples", default_value_t = 0)]
    determinism_samples: usize,

    /// Optional max average inference latency threshold in microseconds.
    #[arg(long = "max-infer-us")]
    max_infer_us: Option<f64>,

    /// Optional max peak VRAM threshold in MiB.
    #[arg(long = "max-vram-mb")]
    max_vram_mb: Option<usize>,

    /// Skip failures when runtime requirements are unavailable.
    #[arg(long = "best-effort", default_value_t = true)]
    best_effort: bool,

    /// Emit structured JSON output to stdout.
    #[arg(long = "json", default_value_t = false)]
    json: bool,

    /// Progress output mode to stderr: auto (TTY only), off, human, jsonl.
    #[arg(long = "progress", value_enum, default_value_t = ProgressArg::Auto)]
    progress: ProgressArg,

    /// Emit progress as JSONL records to stderr.
    #[arg(long = "jsonl", default_value_t = false)]
    jsonl: bool,
}

#[derive(Args, Debug, Clone)]
struct ProbeArgs {
    /// CUDA device ordinal.
    #[arg(short = 'd', long = "device")]
    device: Option<u32>,

    /// Probe all visible CUDA devices.
    #[arg(long = "all", default_value_t = false, conflicts_with = "device")]
    all: bool,

    /// Emit JSON probe output.
    #[arg(long = "json", default_value_t = false)]
    json: bool,
}

#[derive(Args, Debug, Clone)]
struct DevicesArgs {
    /// Emit JSON device listing.
    #[arg(long = "json", default_value_t = false)]
    json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProgressArg {
    Auto,
    Off,
    Human,
    Jsonl,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ProfileArg {
    #[value(name = "dev")]
    Dev,
    #[value(name = "production_strict")]
    ProductionStrict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressMode {
    Off,
    Human,
    Jsonl,
}

#[derive(Debug, Clone)]
struct CudaDeviceInfo {
    ordinal: i32,
    name: String,
    total_mem_bytes: u64,
}

#[derive(Debug, Clone)]
struct ProbeDeviceResult {
    ordinal: i32,
    name: String,
    total_mem_bytes: u64,
    ok: bool,
    vram_current_mb: usize,
    vram_peak_mb: usize,
    error: Option<String>,
}

#[cfg(target_os = "linux")]
type CUresult = i32;
#[cfg(target_os = "linux")]
type CUdevice = i32;

#[cfg(target_os = "linux")]
const CUDA_SUCCESS: CUresult = 0;

#[cfg_attr(target_os = "linux", link(name = "cuda"))]
unsafe extern "C" {
    #[cfg(target_os = "linux")]
    fn cuInit(flags: u32) -> CUresult;
    #[cfg(target_os = "linux")]
    fn cuDriverGetVersion(driver_version: *mut i32) -> CUresult;
    #[cfg(target_os = "linux")]
    fn cuDeviceGetCount(count: *mut i32) -> CUresult;
    #[cfg(target_os = "linux")]
    fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    #[cfg(target_os = "linux")]
    fn cuDeviceGetName(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult;
    #[cfg(target_os = "linux")]
    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
}

#[derive(Debug, Clone, Copy)]
enum BenchmarkEncodeMode {
    Nvenc,
    Skipped,
}

#[derive(Debug)]
struct BenchmarkSummary {
    fps: f64,
    frames: u64,
    elapsed_ms: f64,
    decode_avg_us: f64,
    infer_avg_us: f64,
    encode_avg_us: Option<f64>,
}

impl BenchmarkSummary {
    fn to_json(&self) -> String {
        let encode_field = match self.encode_avg_us {
            Some(v) => json_number(v),
            None => "\"skipped\"".to_string(),
        };

        format!(
            "{{\"schema_version\":{},\"command\":\"benchmark\",\"ok\":true,\"fps\":{},\"frames\":{},\"elapsed_ms\":{},\"stages\":{{\"decode\":{},\"infer\":{},\"encode\":{}}}}}",
            JSON_SCHEMA_VERSION,
            json_number(self.fps),
            self.frames,
            json_number(self.elapsed_ms),
            json_number(self.decode_avg_us),
            json_number(self.infer_avg_us),
            encode_field
        )
    }

    fn to_human(&self) -> String {
        let encode = match self.encode_avg_us {
            Some(v) => format!("{:.3}", v),
            None => "skipped".to_string(),
        };
        format!(
            "benchmark: frames={} fps={:.3} elapsed_ms={:.3} decode_avg_us={:.3} infer_avg_us={:.3} encode_avg_us={}",
            self.frames, self.fps, self.elapsed_ms, self.decode_avg_us, self.infer_avg_us, encode
        )
    }
}

#[derive(Default)]
struct SkipEncoder;

impl FrameEncoder for SkipEncoder {
    fn encode(&mut self, _frame: FrameEnvelope) -> Result<()> {
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

struct ProgressReporter {
    notify: Arc<tokio::sync::Notify>,
    handle: tokio::task::JoinHandle<()>,
}

impl ProgressReporter {
    async fn stop(self) {
        self.notify.notify_waiters();
        let _ = self.handle.await;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProgressSnapshot {
    decoded: u64,
    inferred: u64,
    encoded: u64,
}

const JSON_SCHEMA_VERSION: u32 = 1;

fn main() {
    #[cfg(target_os = "linux")]
    maybe_reexec_with_ort_ld_library_path();
    init_tracing();

    let cli = Cli::parse();
    let json_error_command = match &cli.command {
        Commands::Probe(args) if args.json => Some("probe"),
        Commands::Devices(args) if args.json => Some("devices"),
        Commands::Upscale(args) if args.json => Some("upscale"),
        Commands::Benchmark(args) if args.json => Some("benchmark"),
        Commands::Validate(args) if args.json => Some("validate"),
        _ => None,
    };

    let result = match cli.command {
        Commands::Probe(args) => run_probe(args),
        Commands::Devices(args) => run_devices(args),
        Commands::Upscale(args) => {
            let rt = build_runtime();
            rt.block_on(run_upscale(args))
        }
        Commands::Benchmark(args) => {
            let rt = build_runtime();
            rt.block_on(run_benchmark(args))
        }
        Commands::Validate(args) => {
            let rt = build_runtime();
            rt.block_on(run_validate(args))
        }
    };

    match result {
        Ok(()) => std::process::exit(0),
        Err(err) => {
            if let Some(command) = json_error_command {
                println!("{}", command_error_json(command, &err.to_string()));
            } else {
                tracing::error!(error = %err, code = err.error_code(), "Command failed");
            }
            std::process::exit(err.error_code() as i32);
        }
    }
}

#[cfg(target_os = "linux")]
fn maybe_reexec_with_ort_ld_library_path() {
    let mut args = std::env::args();
    let _bin = args.next();
    let sub = args.next().unwrap_or_default().to_ascii_lowercase();
    if sub != "upscale" && sub != "benchmark" {
        return;
    }
    if std::env::var_os("RAVE_ORT_LD_REEXEC").is_some() {
        return;
    }

    let Some(provider_dir) = resolve_ort_provider_dir_for_loader() else {
        return;
    };

    let on_wsl = is_wsl2();
    let wsl_libcuda = "/usr/lib/wsl/lib/libcuda.so.1";
    let provider_dir_s = provider_dir.to_string_lossy().to_string();
    let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let ld_has_provider = current.split(':').any(|p| p == provider_dir_s);

    let preload_current = std::env::var("LD_PRELOAD").unwrap_or_default();
    let should_preload_wsl_cuda = on_wsl
        && Path::new(wsl_libcuda).is_file()
        && !preload_current.split(':').any(|p| p == wsl_libcuda);

    if ld_has_provider && !should_preload_wsl_cuda {
        return;
    }

    let mut new_ld = provider_dir_s.clone();
    if !current.is_empty() {
        new_ld.push(':');
        new_ld.push_str(&current);
    }

    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.args(std::env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", new_ld)
        .env("RAVE_ORT_LD_REEXEC", "1");
    if should_preload_wsl_cuda {
        let mut new_preload = wsl_libcuda.to_string();
        if !preload_current.is_empty() {
            new_preload.push(':');
            new_preload.push_str(&preload_current);
        }
        cmd.env("LD_PRELOAD", new_preload);
    }
    let err = cmd.exec();
    eprintln!("failed to re-exec with ORT loader path: {err}");
}

#[cfg(target_os = "linux")]
fn resolve_ort_provider_dir_for_loader() -> Option<PathBuf> {
    let mut dirs = Vec::<PathBuf>::new();
    if let Some(dir) = std::env::var_os("ORT_DYLIB_PATH") {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::var_os("ORT_LIB_LOCATION") {
        dirs.push(PathBuf::from(dir));
    }

    if is_wsl2() {
        dirs.extend(ort_cache_dirs_newest_first());
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            dirs.push(dir.to_path_buf());
            dirs.push(dir.join("deps"));
        }
    } else {
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            dirs.push(dir.to_path_buf());
            dirs.push(dir.join("deps"));
        }
        dirs.extend(ort_cache_dirs_newest_first());
    }

    let mut seen = HashSet::<PathBuf>::new();
    for dir in dirs {
        if !seen.insert(dir.clone()) {
            continue;
        }
        let shared = dir.join("libonnxruntime_providers_shared.so");
        let cuda = dir.join("libonnxruntime_providers_cuda.so");
        let trt = dir.join("libonnxruntime_providers_tensorrt.so");
        if shared.is_file() && (cuda.is_file() || trt.is_file()) {
            return Some(dir);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn ort_cache_dirs_newest_first() -> Vec<PathBuf> {
    let mut dirs = Vec::<(u128, PathBuf)>::new();
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let base = PathBuf::from(home).join(".cache/ort.pyke.io/dfbin");
    let Ok(triples) = std::fs::read_dir(base) else {
        return Vec::new();
    };
    for triple in triples.flatten() {
        let triple_path = triple.path();
        if !triple_path.is_dir() {
            continue;
        }
        let Ok(hashes) = std::fs::read_dir(triple_path) else {
            continue;
        };
        for hash in hashes.flatten() {
            let path = hash.path();
            if !path.is_dir() {
                continue;
            }
            let modified = std::fs::metadata(&path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            dirs.push((modified, path));
        }
    }
    dirs.sort_by(|a, b| b.cmp(a));
    dirs.into_iter().map(|(_, path)| path).collect()
}

#[cfg(target_os = "linux")]
fn is_wsl2() -> bool {
    std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.to_ascii_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn mapped_libcuda_path() -> Option<String> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines()
        .filter_map(|line| line.split_whitespace().last())
        .find(|p| p.contains("libcuda.so"))
        .map(|s| s.to_string())
}

#[cfg(target_os = "linux")]
fn has_wsl_driver_loader_conflict() -> bool {
    if !is_wsl2() {
        return false;
    }
    mapped_libcuda_path()
        .map(|p| p.contains("/usr/lib/wsl/drivers/"))
        .unwrap_or(false)
}

#[cfg(not(target_os = "linux"))]
fn has_wsl_driver_loader_conflict() -> bool {
    false
}

fn should_emit_wsl_skip_encode_benchmark_json(err: &EngineError) -> bool {
    #[cfg(target_os = "linux")]
    {
        if !is_wsl2() {
            return false;
        }
        if let EngineError::Pipeline(msg) = err {
            return msg.contains("CUDA_ERROR_OPERATING_SYSTEM (304)") && msg.contains("WSL");
        }
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = err;
        false
    }
}

fn init_tracing() {
    let ansi_enabled = std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_ansi(ansi_enabled)
        .init();
}

fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime")
}

fn run_probe(args: ProbeArgs) -> Result<()> {
    let (driver_version, devices) = enumerate_cuda_devices()?;
    if args.all {
        let mut results = Vec::<ProbeDeviceResult>::new();
        let mut successes = 0usize;
        for dev in &devices {
            let result = match GpuContext::new(dev.ordinal as usize) {
                Ok(ctx) => {
                    let (vram_current, vram_peak) = ctx.vram_usage();
                    successes += 1;
                    ProbeDeviceResult {
                        ordinal: dev.ordinal,
                        name: dev.name.clone(),
                        total_mem_bytes: dev.total_mem_bytes,
                        ok: true,
                        vram_current_mb: vram_current / (1024 * 1024),
                        vram_peak_mb: vram_peak / (1024 * 1024),
                        error: None,
                    }
                }
                Err(err) => ProbeDeviceResult {
                    ordinal: dev.ordinal,
                    name: dev.name.clone(),
                    total_mem_bytes: dev.total_mem_bytes,
                    ok: false,
                    vram_current_mb: 0,
                    vram_peak_mb: 0,
                    error: Some(err.to_string()),
                },
            };
            results.push(result);
        }

        if args.json {
            println!("{}", probe_all_json(driver_version, &results));
        } else {
            println!("probe: all devices");
            println!("cuda_driver_version={driver_version}");
            for row in &results {
                if row.ok {
                    println!(
                        "device={} name={} total_mem_mb={} status=ok vram_current_mb={} vram_peak_mb={}",
                        row.ordinal,
                        row.name,
                        row.total_mem_bytes / (1024 * 1024),
                        row.vram_current_mb,
                        row.vram_peak_mb
                    );
                } else {
                    println!(
                        "device={} name={} total_mem_mb={} status=error error={}",
                        row.ordinal,
                        row.name,
                        row.total_mem_bytes / (1024 * 1024),
                        row.error.as_deref().unwrap_or("unknown")
                    );
                }
            }
        }

        if successes == 0 {
            return Err(EngineError::Pipeline(
                "Probe failed on all visible CUDA devices".into(),
            ));
        }
        return Ok(());
    }

    let device = args.device.unwrap_or(0) as usize;
    let dev_info = devices.iter().find(|d| d.ordinal as usize == device);
    let ctx = GpuContext::new(device)?;
    let (vram_current, vram_peak) = ctx.vram_usage();

    if args.json {
        println!(
            "{}",
            probe_json_single(
                device,
                dev_info.map(|d| d.name.as_str()),
                dev_info.map(|d| d.total_mem_bytes),
                vram_current / (1024 * 1024),
                vram_peak / (1024 * 1024),
                driver_version,
            )
        );
    } else {
        println!("probe: ok");
        println!("cuda_driver_version={driver_version}");
        println!("device={device}");
        if let Some(info) = dev_info {
            println!("name={}", info.name);
            println!("total_mem_mb={}", info.total_mem_bytes / (1024 * 1024));
        }
        println!("vram_current_mb={}", vram_current / (1024 * 1024));
        println!("vram_peak_mb={}", vram_peak / (1024 * 1024));
    }

    Ok(())
}

fn run_devices(args: DevicesArgs) -> Result<()> {
    let (driver_version, devices) = enumerate_cuda_devices()?;
    if args.json {
        println!("{}", devices_json(driver_version, &devices));
    } else {
        println!("devices: {}", devices.len());
        println!("cuda_driver_version={driver_version}");
        for dev in &devices {
            println!(
                "device={} name={} total_mem_mb={}",
                dev.ordinal,
                dev.name,
                dev.total_mem_bytes / (1024 * 1024)
            );
        }
    }
    Ok(())
}

async fn run_upscale(args: UpscaleArgs) -> Result<()> {
    ensure_required_paths(&args.shared)?;

    let resolved = resolve_input(&args.shared)?;
    if args.dry_run {
        if args.json {
            println!(
                "{}",
                upscale_json(
                    true,
                    &args.shared.input,
                    &args.output,
                    &args.shared.model,
                    resolved.codec,
                    resolved.width,
                    resolved.height,
                    resolved.fps_num,
                    resolved.fps_den,
                    0.0,
                    0,
                    0,
                )
            );
        } else {
            println!(
                "dry-run: command=upscale input={} output={} model={} codec={:?} resolution={}x{} fps={}/{}",
                args.shared.input.display(),
                args.output.display(),
                args.shared.model.display(),
                resolved.codec,
                resolved.width,
                resolved.height,
                resolved.fps_num,
                resolved.fps_den
            );
        }
        return Ok(());
    }

    if args.shared.graph.is_some() || args.shared.profile != ProfileArg::Dev {
        return run_upscale_with_graph(args).await;
    }

    let wall_start = Instant::now();
    let setup = prepare_runtime(&args.shared, resolved).await?;
    let decoder = create_decoder(&setup, &args.shared.input)?;
    let encoder = create_nvenc_encoder(
        &setup,
        &args.output,
        args.bitrate.unwrap_or(0),
        resolved.fps_num,
        resolved.fps_den,
    )?;

    let pipeline = UpscalePipeline::new(
        setup.ctx.clone(),
        setup.kernels.clone(),
        PipelineConfig {
            decoded_capacity: setup.decoded_capacity,
            preprocessed_capacity: setup.preprocessed_capacity,
            upscaled_capacity: setup.upscaled_capacity,
            encoder_nv12_pitch: setup.nv12_pitch,
            model_precision: setup.precision,
            enable_profiler: true,
            strict_no_host_copies: matches!(args.shared.profile, ProfileArg::ProductionStrict),
            strict_invariants: matches!(args.shared.profile, ProfileArg::ProductionStrict),
        },
    );
    let progress_mode = resolve_progress_mode(args.progress, args.jsonl);
    let metrics = pipeline.metrics();
    let progress = spawn_progress_reporter("upscale", metrics, progress_mode);

    let run_result = pipeline.run(decoder, setup.backend.clone(), encoder).await;
    if let Some(progress) = progress {
        progress.stop().await;
    }
    let shutdown_result = setup.backend.shutdown().await;
    run_result?;
    shutdown_result?;

    let elapsed = wall_start.elapsed();
    let (vram_current, vram_peak) = setup.ctx.vram_usage();
    tracing::info!(
        elapsed_s = format!("{:.2}", elapsed.as_secs_f64()),
        vram_current_mb = vram_current / (1024 * 1024),
        vram_peak_mb = vram_peak / (1024 * 1024),
        "Upscale complete"
    );
    if args.json {
        println!(
            "{}",
            upscale_json(
                false,
                &args.shared.input,
                &args.output,
                &args.shared.model,
                resolved.codec,
                resolved.width,
                resolved.height,
                resolved.fps_num,
                resolved.fps_den,
                elapsed.as_secs_f64() * 1000.0,
                vram_current / (1024 * 1024),
                vram_peak / (1024 * 1024),
            )
        );
    } else {
        println!(
            "upscale: ok output={} elapsed_s={:.3} vram_peak_mb={}",
            args.output.display(),
            elapsed.as_secs_f64(),
            vram_peak / (1024 * 1024)
        );
    }

    Ok(())
}

async fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    ensure_required_paths(&args.shared)?;

    if args.shared.graph.is_some() {
        return Err(EngineError::Pipeline(
            "benchmark does not support --graph yet; use `rave validate` or `rave upscale --profile ... --graph ...`".into(),
        ));
    }

    let resolved = resolve_input(&args.shared)?;
    let emit_json_stdout = args.json;
    if args.dry_run {
        let summary = BenchmarkSummary {
            fps: 0.0,
            frames: 0,
            elapsed_ms: 0.0,
            decode_avg_us: 0.0,
            infer_avg_us: 0.0,
            encode_avg_us: None,
        };
        emit_benchmark_output(&summary, args.json_out.as_deref(), emit_json_stdout)?;
        return Ok(());
    }

    let setup = match prepare_runtime(&args.shared, resolved).await {
        Ok(setup) => setup,
        Err(err) if args.skip_encode && should_emit_wsl_skip_encode_benchmark_json(&err) => {
            tracing::warn!(
                error = %err,
                "Benchmark runtime init failed under WSL loader conflict; emitting skipped benchmark JSON"
            );
            let summary = BenchmarkSummary {
                fps: 0.0,
                frames: 0,
                elapsed_ms: 0.0,
                decode_avg_us: 0.0,
                infer_avg_us: 0.0,
                encode_avg_us: None,
            };
            emit_benchmark_output(&summary, args.json_out.as_deref(), emit_json_stdout)?;
            return Ok(());
        }
        Err(err) => return Err(err),
    };
    let progress_mode = resolve_progress_mode(args.progress, args.jsonl);
    let mut mode = if args.skip_encode {
        BenchmarkEncodeMode::Skipped
    } else {
        BenchmarkEncodeMode::Nvenc
    };

    let first_result = run_benchmark_once(&setup, &args, mode, progress_mode).await;
    let result = match first_result {
        Ok(summary) => Ok(summary),
        Err(err) if matches!(mode, BenchmarkEncodeMode::Nvenc) && args.allow_encode_skip => {
            if is_nvenc_failure(&err) {
                tracing::warn!(
                    error = %err,
                    "NVENC failed during benchmark; rerunning with encode skipped"
                );
                mode = BenchmarkEncodeMode::Skipped;
                run_benchmark_once(&setup, &args, mode, progress_mode).await
            } else {
                Err(err)
            }
        }
        Err(err) => Err(err),
    };

    let shutdown_result = setup.backend.shutdown().await;
    let summary = result?;
    shutdown_result?;

    emit_benchmark_output(&summary, args.json_out.as_deref(), emit_json_stdout)?;
    Ok(())
}

async fn run_upscale_with_graph(args: UpscaleArgs) -> Result<()> {
    let wall_start = Instant::now();
    let resolved = resolve_input(&args.shared)?;
    let profile = profile_to_preset(args.shared.profile);
    let graph = load_stage_graph(
        args.shared.graph.as_deref(),
        Some(&args.shared.model),
        args.shared.precision.as_deref(),
    )?;

    let device = args.shared.device.unwrap_or(0) as usize;
    let (ctx, kernels) = pipeline_create_context_and_kernels(
        device,
        args.shared.vram_limit,
        matches!(args.shared.profile, ProfileArg::ProductionStrict),
    )?;

    let pipeline = UpscalePipeline::new(
        ctx.clone(),
        kernels,
        PipelineConfig {
            decoded_capacity: args.shared.decode_cap.unwrap_or(4),
            preprocessed_capacity: args.shared.preprocess_cap.unwrap_or(2),
            upscaled_capacity: args.shared.upscale_cap.unwrap_or(4),
            encoder_nv12_pitch: 256,
            model_precision: pipeline_parse_precision(
                args.shared.precision.as_deref().unwrap_or("fp32"),
            )?,
            enable_profiler: true,
            strict_no_host_copies: false,
            strict_invariants: matches!(args.shared.profile, ProfileArg::ProductionStrict),
        },
    );

    let mut contract = RunContract::for_profile(profile);
    contract.requested_device = device as u32;
    let report = pipeline
        .run_graph(&args.shared.input, &args.output, graph, profile, contract)
        .await?;

    let elapsed = wall_start.elapsed();
    if args.json {
        println!(
            "{}",
            upscale_json(
                false,
                &args.shared.input,
                &args.output,
                &args.shared.model,
                resolved.codec,
                resolved.width,
                resolved.height,
                resolved.fps_num,
                resolved.fps_den,
                elapsed.as_secs_f64() * 1000.0,
                report.vram_current_bytes / (1024 * 1024),
                report.vram_peak_bytes / (1024 * 1024),
            )
        );
    } else {
        println!(
            "upscale: ok output={} elapsed_s={:.3} vram_peak_mb={} profile={}",
            args.output.display(),
            elapsed.as_secs_f64(),
            report.vram_peak_bytes / (1024 * 1024),
            profile_label(args.shared.profile)
        );
    }

    Ok(())
}

#[derive(Debug)]
struct ValidateSummary {
    ok: bool,
    skipped: bool,
    profile: ProfileArg,
    model_path: Option<String>,
    runs: u32,
    determinism_equal: Option<bool>,
    max_infer_us: f64,
    peak_vram_mb: usize,
    frames_decoded: u64,
    frames_encoded: u64,
    warnings: Vec<String>,
}

impl ValidateSummary {
    fn to_human(&self) -> String {
        let model = self.model_path.as_deref().unwrap_or("n/a");
        if self.skipped {
            return format!(
                "validate: skipped profile={} model={} warnings={}",
                profile_label(self.profile),
                model,
                self.warnings.join(" | ")
            );
        }

        let determinism = self
            .determinism_equal
            .map(|v| if v { "pass" } else { "fail" })
            .unwrap_or("n/a");
        format!(
            "validate: ok={} profile={} model={} runs={} determinism={} max_infer_us={:.3} peak_vram_mb={} frames_decoded={} frames_encoded={} warnings={}",
            self.ok,
            profile_label(self.profile),
            model,
            self.runs,
            determinism,
            self.max_infer_us,
            self.peak_vram_mb,
            self.frames_decoded,
            self.frames_encoded,
            self.warnings.join(" | ")
        )
    }

    fn to_json(&self) -> String {
        let warnings = self
            .warnings
            .iter()
            .map(|v| json_string(v))
            .collect::<Vec<_>>()
            .join(",");
        let determinism = self
            .determinism_equal
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        let model_path = self
            .model_path
            .as_ref()
            .map(|path| json_string(path))
            .unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"schema_version\":{},\"command\":\"validate\",\"ok\":{},\"skipped\":{},\"profile\":{},\"model_path\":{},\"runs\":{},\"determinism_equal\":{},\"max_infer_us\":{},\"peak_vram_mb\":{},\"frames\":{{\"decoded\":{},\"encoded\":{}}},\"warnings\":[{}]}}",
            JSON_SCHEMA_VERSION,
            self.ok,
            self.skipped,
            json_string(profile_label(self.profile)),
            model_path,
            self.runs,
            determinism,
            json_number(self.max_infer_us),
            self.peak_vram_mb,
            self.frames_decoded,
            self.frames_encoded,
            warnings
        )
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ValidateFixture {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    graph: Option<PathBuf>,
    model: Option<PathBuf>,
    model_env: Option<String>,
    profile: Option<String>,
    determinism_runs: Option<u32>,
    determinism_samples: Option<usize>,
    max_infer_us: Option<f64>,
    max_vram_mb: Option<usize>,
    allow_hash_skipped: Option<bool>,
}

impl ValidateFixture {
    fn resolve_paths(mut self, base: &Path) -> Self {
        self.input = self.input.map(|path| resolve_fixture_path(base, path));
        self.output = self.output.map(|path| resolve_fixture_path(base, path));
        self.graph = self.graph.map(|path| resolve_fixture_path(base, path));
        self.model = self.model.map(|path| resolve_fixture_path(base, path));
        self
    }
}

fn resolve_fixture_path(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn load_validate_fixture(path: &Path) -> Result<ValidateFixture> {
    let data = std::fs::read_to_string(path).map_err(|err| {
        EngineError::Pipeline(format!(
            "Failed to read validate fixture {}: {err}",
            path.display()
        ))
    })?;
    let fixture = serde_json::from_str::<ValidateFixture>(&data).map_err(|err| {
        EngineError::Pipeline(format!(
            "Invalid validate fixture JSON {}: {err}",
            path.display()
        ))
    })?;
    let base = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(fixture.resolve_paths(base))
}

fn parse_profile_arg(raw: &str) -> Result<ProfileArg> {
    match raw.to_ascii_lowercase().as_str() {
        "dev" => Ok(ProfileArg::Dev),
        "production_strict" => Ok(ProfileArg::ProductionStrict),
        other => Err(EngineError::Pipeline(format!(
            "Unknown validate fixture profile '{other}'. Use dev or production_strict."
        ))),
    }
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

const DEFAULT_VALIDATE_MODEL_RELATIVE: &str = "tests/assets/models/resize2x_rgb.onnx";

fn default_validate_model_path() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap_or(manifest_dir);
    workspace_root.join(DEFAULT_VALIDATE_MODEL_RELATIVE)
}

fn resolve_fixture_model_path_with_default_and_lookup<F>(
    fixture: &ValidateFixture,
    default_model: &Path,
    env_lookup: F,
) -> Result<Option<PathBuf>>
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(path) = &fixture.model {
        return Ok(Some(path.clone()));
    }
    if let Some(env_name) = &fixture.model_env {
        if let Some(value) = env_lookup(env_name) {
            return Ok(Some(PathBuf::from(value)));
        }
        if env_name == "RAVE_VALIDATE_MODEL" {
            if default_model.exists() {
                return Ok(Some(default_model.to_path_buf()));
            }
            return Err(EngineError::Pipeline(format!(
                "Validate fixture default model not found: {}. Set {} to override.",
                default_model.display(),
                env_name
            )));
        }
        return Err(EngineError::Pipeline(format!(
            "Validate fixture requires env var {} to resolve model path",
            env_name
        )));
    }
    Ok(None)
}

fn resolve_fixture_model_path(fixture: &ValidateFixture) -> Result<Option<PathBuf>> {
    let fallback = default_validate_model_path();
    resolve_fixture_model_path_with_default_and_lookup(fixture, &fallback, |name| {
        std::env::var(name).ok()
    })
}

fn emit_validate_skip(
    profile: ProfileArg,
    json: bool,
    progress_mode: ProgressMode,
    runs: u32,
    model_path: Option<&Path>,
    warning: String,
) {
    let summary = ValidateSummary {
        ok: true,
        skipped: true,
        profile,
        model_path: model_path.map(|path| path.display().to_string()),
        runs,
        determinism_equal: None,
        max_infer_us: 0.0,
        peak_vram_mb: 0,
        frames_decoded: 0,
        frames_encoded: 0,
        warnings: vec![warning],
    };
    if json {
        println!("{}", summary.to_json());
    } else {
        println!("{}", summary.to_human());
    }
    emit_progress_line(
        "validate",
        progress_mode,
        Duration::from_millis(0),
        ProgressSnapshot {
            decoded: 0,
            inferred: 0,
            encoded: 0,
        },
        true,
    );
}

async fn run_validate(args: ValidateArgs) -> Result<()> {
    let progress_mode = resolve_progress_mode(args.progress, args.jsonl);
    emit_progress_line(
        "validate",
        progress_mode,
        Duration::from_millis(0),
        ProgressSnapshot {
            decoded: 0,
            inferred: 0,
            encoded: 0,
        },
        false,
    );

    let fixture = args
        .fixture
        .as_deref()
        .map(load_validate_fixture)
        .transpose()?;
    let ci_gpu_mode = env_flag_enabled("RAVE_CI_GPU");
    let best_effort = if ci_gpu_mode { false } else { args.best_effort };

    let mut profile_arg = args.profile;
    if matches!(profile_arg, ProfileArg::ProductionStrict)
        && let Some(raw_profile) = fixture.as_ref().and_then(|f| f.profile.as_deref())
    {
        profile_arg = parse_profile_arg(raw_profile)?;
    }
    let profile = profile_to_preset(profile_arg);

    let input = args
        .input
        .clone()
        .or_else(|| fixture.as_ref().and_then(|f| f.input.clone()));
    let Some(input_path) = input else {
        emit_validate_skip(
            profile_arg,
            args.json,
            progress_mode,
            0,
            None,
            "no input provided; validation skipped".into(),
        );
        return Ok(());
    };

    let fixture_model = match fixture.as_ref().map(resolve_fixture_model_path).transpose() {
        Ok(model) => model.flatten(),
        Err(err) if best_effort => {
            emit_validate_skip(
                profile_arg,
                args.json,
                progress_mode,
                0,
                None,
                err.to_string(),
            );
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let graph_path = args
        .graph
        .clone()
        .or_else(|| fixture.as_ref().and_then(|f| f.graph.clone()));
    let model = args.model.clone().or(fixture_model);
    let resolved_model_path = model.clone();
    let mut graph = match load_stage_graph(graph_path.as_deref(), model.as_ref(), None) {
        Ok(graph) => graph,
        Err(err) if best_effort => {
            emit_validate_skip(
                profile_arg,
                args.json,
                progress_mode,
                0,
                resolved_model_path.as_deref(),
                format!("validation graph unavailable: {err}"),
            );
            return Ok(());
        }
        Err(err) => return Err(err),
    };
    if graph_path.is_some()
        && let Some(model_path) = model.as_deref()
    {
        override_enhance_model_path(&mut graph, model_path);
    }

    let device = args.device.unwrap_or(0) as usize;
    let output_path = args
        .output
        .clone()
        .or_else(|| fixture.as_ref().and_then(|f| f.output.clone()))
        .unwrap_or_else(|| std::env::temp_dir().join("rave_validate_out.mp4"));

    let mut determinism_runs = args.determinism_runs;
    if determinism_runs == 2
        && let Some(value) = fixture.as_ref().and_then(|f| f.determinism_runs)
    {
        determinism_runs = value;
    }
    let mut determinism_samples = args.determinism_samples;
    if determinism_samples == 0
        && let Some(value) = fixture.as_ref().and_then(|f| f.determinism_samples)
    {
        determinism_samples = value;
    }
    let max_infer_us = args
        .max_infer_us
        .or_else(|| fixture.as_ref().and_then(|f| f.max_infer_us));
    let max_vram_mb = args
        .max_vram_mb
        .or_else(|| fixture.as_ref().and_then(|f| f.max_vram_mb));
    let allow_hash_skipped = fixture
        .as_ref()
        .and_then(|f| f.allow_hash_skipped)
        .unwrap_or(true);

    let (ctx, kernels) = match pipeline_create_context_and_kernels(
        device,
        None,
        matches!(profile_arg, ProfileArg::ProductionStrict),
    ) {
        Ok(resources) => resources,
        Err(err) if best_effort => {
            emit_validate_skip(
                profile_arg,
                args.json,
                progress_mode,
                0,
                resolved_model_path.as_deref(),
                format!("runtime unavailable: {err}"),
            );
            return Ok(());
        }
        Err(err) => return Err(err),
    };
    let pipeline = UpscalePipeline::new(
        ctx,
        kernels,
        PipelineConfig {
            decoded_capacity: 4,
            preprocessed_capacity: 2,
            upscaled_capacity: 4,
            encoder_nv12_pitch: 256,
            model_precision: pipeline_parse_precision("fp16")?,
            enable_profiler: true,
            strict_no_host_copies: false,
            strict_invariants: matches!(profile_arg, ProfileArg::ProductionStrict),
        },
    );

    let mut contract = RunContract::for_profile(profile);
    contract.requested_device = device as u32;
    contract.determinism_hash_frames = determinism_samples;
    if determinism_samples > 0 && allow_hash_skipped {
        contract.fail_on_audit_warn = false;
    }
    let runs = if matches!(profile_arg, ProfileArg::ProductionStrict) {
        determinism_runs.max(1)
    } else {
        1
    };

    let mut reports = Vec::<PipelineReport>::new();
    for idx in 0..runs {
        let run_output = if runs > 1 {
            let stem = output_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("rave_validate");
            let ext = output_path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("mp4");
            output_path.with_file_name(format!("{stem}_run{idx}.{ext}"))
        } else {
            output_path.clone()
        };

        let report = match pipeline
            .run_graph(
                &input_path,
                &run_output,
                graph.clone(),
                profile,
                contract.clone(),
            )
            .await
        {
            Ok(report) => report,
            Err(err) if best_effort => {
                emit_validate_skip(
                    profile_arg,
                    args.json,
                    progress_mode,
                    idx,
                    resolved_model_path.as_deref(),
                    format!("validation execution skipped: {err}"),
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };
        reports.push(report);
    }

    let first = reports
        .first()
        .ok_or_else(|| EngineError::Pipeline("validate produced no reports".into()))?;
    let infer_avg = if first.frames_encoded > 0 {
        first.stage_timing.infer_us as f64 / first.frames_encoded as f64
    } else {
        0.0
    };
    let peak_vram_mb = first.vram_peak_bytes / (1024 * 1024);

    if let Some(limit) = max_infer_us
        && infer_avg > limit
    {
        return Err(EngineError::InvariantViolation(format!(
            "validate threshold failed: infer_avg_us={infer_avg:.3} > {limit:.3}"
        )));
    }
    if let Some(limit) = max_vram_mb
        && peak_vram_mb > limit
    {
        return Err(EngineError::InvariantViolation(format!(
            "validate threshold failed: peak_vram_mb={peak_vram_mb} > {limit}"
        )));
    }
    if first.frames_decoded != first.frames_encoded {
        return Err(EngineError::InvariantViolation(format!(
            "validate failed: frame count mismatch decoded={} encoded={}",
            first.frames_decoded, first.frames_encoded
        )));
    }

    let determinism_requested = contract.determinism_hash_frames > 0;
    let hashes_available = reports
        .iter()
        .all(|report| !report.stage_checksums.is_empty());
    let determinism_equal = if runs > 1 && contract.deterministic_output && hashes_available {
        let baseline = &reports[0].stage_checksums;
        Some(
            reports
                .iter()
                .skip(1)
                .all(|report| report.stage_checksums == *baseline),
        )
    } else {
        None
    };
    if determinism_requested && !hashes_available && !allow_hash_skipped {
        return Err(EngineError::InvariantViolation(
            "validate determinism failed: checkpoint hashes were requested but unavailable".into(),
        ));
    }
    if determinism_equal == Some(false) {
        return Err(EngineError::InvariantViolation(
            "validate determinism failed: stage checksums differ across runs".into(),
        ));
    }

    let summary = ValidateSummary {
        ok: true,
        skipped: false,
        profile: profile_arg,
        model_path: resolved_model_path
            .as_ref()
            .map(|path| path.display().to_string()),
        runs,
        determinism_equal,
        max_infer_us: infer_avg,
        peak_vram_mb,
        frames_decoded: first.frames_decoded,
        frames_encoded: first.frames_encoded,
        warnings: reports
            .iter()
            .flat_map(|report| report.audit.iter())
            .filter(|check| matches!(check.level, AuditLevel::Warn))
            .map(|check| format!("{}: {}", check.code, check.message))
            .collect(),
    };

    if args.json {
        println!("{}", summary.to_json());
    } else {
        println!("{}", summary.to_human());
    }

    emit_progress_line(
        "validate",
        progress_mode,
        Duration::from_millis(0),
        ProgressSnapshot {
            decoded: first.frames_decoded,
            inferred: first.frames_encoded,
            encoded: first.frames_encoded,
        },
        true,
    );
    Ok(())
}

async fn run_benchmark_once(
    setup: &RuntimeSetup,
    args: &BenchmarkArgs,
    mode: BenchmarkEncodeMode,
    progress_mode: ProgressMode,
) -> Result<BenchmarkSummary> {
    if matches!(mode, BenchmarkEncodeMode::Skipped) && has_wsl_driver_loader_conflict() {
        tracing::warn!(
            "Detected /usr/lib/wsl/drivers libcuda loader conflict; skipping NVDEC benchmark path to avoid segfault. \
Run with: LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}"
        );
        return Ok(BenchmarkSummary {
            fps: 0.0,
            frames: 0,
            elapsed_ms: 0.0,
            decode_avg_us: 0.0,
            infer_avg_us: 0.0,
            encode_avg_us: None,
        });
    }

    let decoder = create_decoder(setup, &args.shared.input)?;
    let pipeline = UpscalePipeline::new(
        setup.ctx.clone(),
        setup.kernels.clone(),
        PipelineConfig {
            decoded_capacity: setup.decoded_capacity,
            preprocessed_capacity: setup.preprocessed_capacity,
            upscaled_capacity: setup.upscaled_capacity,
            encoder_nv12_pitch: setup.nv12_pitch,
            model_precision: setup.precision,
            enable_profiler: true,
            strict_no_host_copies: false,
            strict_invariants: matches!(args.shared.profile, ProfileArg::ProductionStrict),
        },
    );
    let metrics = pipeline.metrics();
    let progress = spawn_progress_reporter("benchmark", metrics.clone(), progress_mode);

    let start = Instant::now();
    match mode {
        BenchmarkEncodeMode::Nvenc => {
            tracing::info!("Benchmark encode path: NVENC enabled");
            let out = benchmark_output_path(args);
            let encoder = create_nvenc_encoder(
                setup,
                &out,
                args.bitrate.unwrap_or(0),
                setup.input.fps_num,
                setup.input.fps_den,
            )?;
            pipeline
                .run(decoder, setup.backend.clone(), encoder)
                .await?;
        }
        BenchmarkEncodeMode::Skipped => {
            tracing::info!("Benchmark encode path: skipped (--skip-encode)");
            let encoder = SkipEncoder;
            pipeline
                .run(decoder, setup.backend.clone(), encoder)
                .await?;
        }
    }
    let elapsed = start.elapsed();
    if let Some(progress) = progress {
        progress.stop().await;
    }

    Ok(make_benchmark_summary(
        &metrics,
        elapsed,
        matches!(mode, BenchmarkEncodeMode::Skipped),
    ))
}

fn emit_benchmark_output(
    summary: &BenchmarkSummary,
    json_out: Option<&Path>,
    emit_json_stdout: bool,
) -> Result<()> {
    let json = summary.to_json();
    if emit_json_stdout {
        println!("{json}");
    } else {
        println!("{}", summary.to_human());
    }

    if let Some(path) = json_out {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                EngineError::Pipeline(format!(
                    "Failed to create benchmark output directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        std::fs::write(path, &json).map_err(|e| {
            EngineError::Pipeline(format!(
                "Failed to write benchmark JSON to {}: {e}",
                path.display()
            ))
        })?;
    }

    Ok(())
}

fn make_benchmark_summary(
    metrics: &PipelineMetrics,
    elapsed: Duration,
    encode_skipped: bool,
) -> BenchmarkSummary {
    let decoded = metrics
        .frames_decoded
        .load(std::sync::atomic::Ordering::Relaxed);
    let inferred = metrics
        .frames_inferred
        .load(std::sync::atomic::Ordering::Relaxed);
    let encoded = metrics
        .frames_encoded
        .load(std::sync::atomic::Ordering::Relaxed);

    let measured_frames = if encode_skipped {
        inferred
    } else {
        encoded.max(inferred)
    };
    let elapsed_s = elapsed.as_secs_f64();

    BenchmarkSummary {
        fps: if measured_frames > 0 && elapsed_s > 0.0 {
            measured_frames as f64 / elapsed_s
        } else {
            0.0
        },
        frames: measured_frames,
        elapsed_ms: elapsed_s * 1000.0,
        decode_avg_us: avg_us(
            metrics
                .decode_total_us
                .load(std::sync::atomic::Ordering::Relaxed),
            decoded,
        ),
        infer_avg_us: avg_us(
            metrics
                .inference_total_us
                .load(std::sync::atomic::Ordering::Relaxed),
            inferred,
        ),
        encode_avg_us: if encode_skipped {
            None
        } else {
            Some(avg_us(
                metrics
                    .encode_total_us
                    .load(std::sync::atomic::Ordering::Relaxed),
                encoded,
            ))
        },
    }
}

fn resolve_progress_mode(progress: ProgressArg, jsonl: bool) -> ProgressMode {
    if jsonl {
        return ProgressMode::Jsonl;
    }
    match progress {
        ProgressArg::Auto => {
            if std::io::stderr().is_terminal() {
                ProgressMode::Human
            } else {
                ProgressMode::Off
            }
        }
        ProgressArg::Off => ProgressMode::Off,
        ProgressArg::Human => ProgressMode::Human,
        ProgressArg::Jsonl => ProgressMode::Jsonl,
    }
}

fn current_progress_snapshot(metrics: &PipelineMetrics) -> ProgressSnapshot {
    ProgressSnapshot {
        decoded: metrics.frames_decoded.load(Ordering::Relaxed),
        inferred: metrics.frames_inferred.load(Ordering::Relaxed),
        encoded: metrics.frames_encoded.load(Ordering::Relaxed),
    }
}

fn emit_progress_line(
    command: &'static str,
    mode: ProgressMode,
    elapsed: Duration,
    snapshot: ProgressSnapshot,
    final_line: bool,
) {
    // Progress contract: progress events are written to stderr only.
    match mode {
        ProgressMode::Off => {}
        ProgressMode::Human => {
            eprintln!(
                "progress: command={} elapsed_s={:.3} decoded={} inferred={} encoded={} final={}",
                command,
                elapsed.as_secs_f64(),
                snapshot.decoded,
                snapshot.inferred,
                snapshot.encoded,
                final_line
            );
        }
        ProgressMode::Jsonl => {
            eprintln!(
                "{{\"schema_version\":{},\"type\":\"progress\",\"command\":{},\"elapsed_ms\":{},\"frames\":{{\"decoded\":{},\"inferred\":{},\"encoded\":{}}},\"final\":{}}}",
                JSON_SCHEMA_VERSION,
                json_string(command),
                elapsed.as_millis(),
                snapshot.decoded,
                snapshot.inferred,
                snapshot.encoded,
                final_line
            );
        }
    }
}

fn spawn_progress_reporter(
    command: &'static str,
    metrics: Arc<PipelineMetrics>,
    mode: ProgressMode,
) -> Option<ProgressReporter> {
    if matches!(mode, ProgressMode::Off) {
        return None;
    }

    let notify = Arc::new(tokio::sync::Notify::new());
    let notify_task = notify.clone();
    let handle = tokio::spawn(async move {
        let start = Instant::now();
        let mut last = current_progress_snapshot(&metrics);
        loop {
            tokio::select! {
                _ = notify_task.notified() => {
                    let snapshot = current_progress_snapshot(&metrics);
                    emit_progress_line(command, mode, start.elapsed(), snapshot, true);
                    break;
                }
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    let snapshot = current_progress_snapshot(&metrics);
                    if snapshot != last {
                        emit_progress_line(command, mode, start.elapsed(), snapshot, false);
                        last = snapshot;
                    }
                }
            }
        }
    });

    Some(ProgressReporter { notify, handle })
}

fn avg_us(total_us: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total_us as f64 / count as f64
    }
}

fn json_number(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "0.0".to_string()
    }
}

fn json_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

fn command_error_json(command: &str, error: &str) -> String {
    format!(
        "{{\"schema_version\":{},\"command\":{},\"ok\":false,\"error\":{}}}",
        JSON_SCHEMA_VERSION,
        json_string(command),
        json_string(error)
    )
}

fn probe_json_single(
    device: usize,
    name: Option<&str>,
    total_mem_bytes: Option<u64>,
    vram_current_mb: usize,
    vram_peak_mb: usize,
    driver_version: i32,
) -> String {
    let name_field = name
        .map(json_string)
        .unwrap_or_else(|| "\"unknown\"".to_string());
    let total_mem_mb = total_mem_bytes.unwrap_or(0) / (1024 * 1024);
    format!(
        "{{\"schema_version\":{},\"command\":\"probe\",\"ok\":true,\"device\":{},\"name\":{},\"total_mem_mb\":{},\"cuda_driver_version\":{},\"vram_current_mb\":{},\"vram_peak_mb\":{}}}",
        JSON_SCHEMA_VERSION,
        device,
        name_field,
        total_mem_mb,
        driver_version,
        vram_current_mb,
        vram_peak_mb
    )
}

#[allow(clippy::too_many_arguments)]
fn upscale_json(
    dry_run: bool,
    input: &Path,
    output: &Path,
    model: &Path,
    codec: cudaVideoCodec,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    elapsed_ms: f64,
    vram_current_mb: usize,
    vram_peak_mb: usize,
) -> String {
    format!(
        "{{\"schema_version\":{},\"command\":\"upscale\",\"ok\":true,\"dry_run\":{},\"input\":{},\"output\":{},\"model\":{},\"codec\":{},\"width\":{},\"height\":{},\"fps_num\":{},\"fps_den\":{},\"elapsed_ms\":{},\"vram_current_mb\":{},\"vram_peak_mb\":{}}}",
        JSON_SCHEMA_VERSION,
        dry_run,
        json_string(&input.display().to_string()),
        json_string(&output.display().to_string()),
        json_string(&model.display().to_string()),
        json_string(&format!("{codec:?}")),
        width,
        height,
        fps_num,
        fps_den,
        json_number(elapsed_ms),
        vram_current_mb,
        vram_peak_mb
    )
}

fn devices_json(driver_version: i32, devices: &[CudaDeviceInfo]) -> String {
    let entries = devices
        .iter()
        .map(|d| {
            format!(
                "{{\"device\":{},\"name\":{},\"total_mem_mb\":{}}}",
                d.ordinal,
                json_string(&d.name),
                d.total_mem_bytes / (1024 * 1024)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema_version\":{},\"command\":\"devices\",\"ok\":true,\"cuda_driver_version\":{},\"devices\":[{}]}}",
        JSON_SCHEMA_VERSION, driver_version, entries
    )
}

fn probe_all_json(driver_version: i32, rows: &[ProbeDeviceResult]) -> String {
    let ok = rows.iter().any(|r| r.ok);
    let entries = rows
        .iter()
        .map(|r| {
            let err = r
                .error
                .as_deref()
                .map(json_string)
                .unwrap_or_else(|| "null".to_string());
            format!(
                "{{\"device\":{},\"name\":{},\"total_mem_mb\":{},\"ok\":{},\"vram_current_mb\":{},\"vram_peak_mb\":{},\"error\":{}}}",
                r.ordinal,
                json_string(&r.name),
                r.total_mem_bytes / (1024 * 1024),
                r.ok,
                r.vram_current_mb,
                r.vram_peak_mb,
                err
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema_version\":{},\"command\":\"probe\",\"ok\":{},\"all\":true,\"cuda_driver_version\":{},\"results\":[{}]}}",
        JSON_SCHEMA_VERSION, ok, driver_version, entries
    )
}

#[cfg(target_os = "linux")]
fn enumerate_cuda_devices() -> Result<(i32, Vec<CudaDeviceInfo>)> {
    // SAFETY: CUDA Driver API call with constant flags.
    let rc_init = unsafe { cuInit(0) };
    if rc_init != CUDA_SUCCESS {
        return Err(EngineError::Pipeline(format!(
            "cuInit failed with rc={rc_init}; run scripts/run_cuda_probe.sh for diagnostics"
        )));
    }

    let mut driver_version = -1i32;
    // SAFETY: valid out pointer.
    let rc_drv = unsafe { cuDriverGetVersion(&mut driver_version as *mut i32) };
    if rc_drv != CUDA_SUCCESS {
        return Err(EngineError::Pipeline(format!(
            "cuDriverGetVersion failed with rc={rc_drv}"
        )));
    }

    let mut count = 0i32;
    // SAFETY: valid out pointer.
    let rc_count = unsafe { cuDeviceGetCount(&mut count as *mut i32) };
    if rc_count != CUDA_SUCCESS {
        return Err(EngineError::Pipeline(format!(
            "cuDeviceGetCount failed with rc={rc_count}"
        )));
    }

    let mut devices = Vec::<CudaDeviceInfo>::new();
    for ordinal in 0..count {
        let mut dev = -1i32;
        // SAFETY: valid out pointer and ordinal within declared range.
        let rc_get = unsafe { cuDeviceGet(&mut dev as *mut CUdevice, ordinal) };
        if rc_get != CUDA_SUCCESS {
            return Err(EngineError::Pipeline(format!(
                "cuDeviceGet({ordinal}) failed with rc={rc_get}"
            )));
        }

        let mut name_buf = [0 as c_char; 100];
        // SAFETY: valid output buffer and device handle.
        let rc_name = unsafe { cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, dev) };
        if rc_name != CUDA_SUCCESS {
            return Err(EngineError::Pipeline(format!(
                "cuDeviceGetName({ordinal}) failed with rc={rc_name}"
            )));
        }
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_string();

        let mut total_mem = 0usize;
        // SAFETY: valid out pointer and device handle.
        let rc_mem = unsafe { cuDeviceTotalMem_v2(&mut total_mem as *mut usize, dev) };
        if rc_mem != CUDA_SUCCESS {
            return Err(EngineError::Pipeline(format!(
                "cuDeviceTotalMem_v2({ordinal}) failed with rc={rc_mem}"
            )));
        }

        devices.push(CudaDeviceInfo {
            ordinal,
            name,
            total_mem_bytes: total_mem as u64,
        });
    }
    Ok((driver_version, devices))
}

#[cfg(not(target_os = "linux"))]
fn enumerate_cuda_devices() -> Result<(i32, Vec<CudaDeviceInfo>)> {
    Err(EngineError::Pipeline(
        "Device enumeration is currently supported on Linux only".into(),
    ))
}

fn is_nvenc_failure(err: &EngineError) -> bool {
    let message = err.to_string().to_ascii_lowercase();
    message.contains("nvenc")
        || message.contains("nvidia-encode")
        || message.contains("encode session")
}

fn benchmark_output_path(args: &BenchmarkArgs) -> PathBuf {
    args.output
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("rave_benchmark_out.265"))
}

async fn prepare_runtime(shared: &SharedVideoArgs, input: ResolvedInput) -> Result<RuntimeSetup> {
    let device = shared.device.unwrap_or(0) as usize;
    pipeline_prepare_runtime(&RuntimeRequest {
        model_path: shared.model.clone(),
        precision: shared
            .precision
            .clone()
            .unwrap_or_else(|| "fp32".to_string()),
        device,
        vram_limit_mib: shared.vram_limit,
        strict_vram_limit: matches!(shared.profile, ProfileArg::ProductionStrict),
        decode_cap: shared.decode_cap,
        preprocess_cap: shared.preprocess_cap,
        upscale_cap: shared.upscale_cap,
        resolved_input: input,
    })
    .await
}

fn ensure_required_paths(args: &SharedVideoArgs) -> Result<()> {
    if !args.input.exists() {
        return Err(EngineError::Pipeline(format!(
            "Input file not found: {}",
            args.input.display()
        )));
    }
    if !args.model.exists() {
        return Err(EngineError::Pipeline(format!(
            "Model file not found: {}",
            args.model.display()
        )));
    }
    Ok(())
}

fn resolve_input(shared: &SharedVideoArgs) -> Result<ResolvedInput> {
    pipeline_resolve_input(
        &shared.input,
        shared.codec.as_deref(),
        shared.width,
        shared.height,
        shared.fps_num,
        shared.fps_den,
    )
}

fn create_decoder(setup: &RuntimeSetup, input_path: &Path) -> Result<PipelineDecoder> {
    pipeline_create_decoder(setup, input_path)
}

fn create_nvenc_encoder(
    setup: &RuntimeSetup,
    output_path: &Path,
    bitrate_kbps: u32,
    fps_num: u32,
    fps_den: u32,
) -> Result<PipelineEncoder> {
    pipeline_create_nvenc_encoder(setup, output_path, bitrate_kbps, fps_num, fps_den)
}

fn parse_precision_policy(s: &str) -> Result<PrecisionPolicyConfig> {
    match s.to_ascii_lowercase().as_str() {
        "fp32" | "f32" | "float32" => Ok(PrecisionPolicyConfig::Fp32),
        "fp16" | "f16" | "float16" | "half" => Ok(PrecisionPolicyConfig::Fp16),
        other => Err(EngineError::Pipeline(format!(
            "Unknown precision policy '{other}'. Use fp32 or fp16."
        ))),
    }
}

fn profile_to_preset(profile: ProfileArg) -> ProfilePreset {
    match profile {
        ProfileArg::Dev => ProfilePreset::Dev,
        ProfileArg::ProductionStrict => ProfilePreset::ProductionStrict,
    }
}

fn profile_label(profile: ProfileArg) -> &'static str {
    match profile {
        ProfileArg::Dev => "dev",
        ProfileArg::ProductionStrict => "production_strict",
    }
}

fn load_stage_graph(
    graph_path: Option<&Path>,
    model: Option<&PathBuf>,
    precision: Option<&str>,
) -> Result<StageGraph> {
    let graph = if let Some(path) = graph_path {
        StageGraph::from_json_file(path)?
    } else {
        let model_path = model
            .ok_or_else(|| {
                EngineError::Pipeline(
                    "Graph not provided and model path missing for default Enhance stage".into(),
                )
            })?
            .clone();

        StageGraph {
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            stages: vec![StageConfig::Enhance {
                id: StageId(1),
                config: EnhanceConfig {
                    model_path,
                    precision_policy: parse_precision_policy(precision.unwrap_or("fp16"))?,
                    batch_config: GraphBatchConfig::default(),
                    scale: 2,
                },
            }],
        }
    };

    graph.validate()?;
    Ok(graph)
}

fn override_enhance_model_path(graph: &mut StageGraph, model_path: &Path) {
    for stage in &mut graph.stages {
        if let StageConfig::Enhance { config, .. } = stage {
            config.model_path = model_path.to_path_buf();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_model_env_override_takes_precedence() {
        let fixture = ValidateFixture {
            model_env: Some("RAVE_VALIDATE_MODEL".into()),
            ..ValidateFixture::default()
        };
        let resolved = resolve_fixture_model_path_with_default_and_lookup(
            &fixture,
            Path::new("/tmp/default_resize2x.onnx"),
            |name| {
                if name == "RAVE_VALIDATE_MODEL" {
                    Some("/tmp/override_model.onnx".into())
                } else {
                    None
                }
            },
        )
        .expect("env override should resolve");

        assert_eq!(resolved, Some(PathBuf::from("/tmp/override_model.onnx")));
    }

    #[test]
    fn fixture_model_env_falls_back_to_committed_model() {
        let fixture = ValidateFixture {
            model_env: Some("RAVE_VALIDATE_MODEL".into()),
            ..ValidateFixture::default()
        };
        let committed = default_validate_model_path();
        assert!(
            committed.exists(),
            "committed validate model is missing at {}",
            committed.display()
        );

        let resolved =
            resolve_fixture_model_path_with_default_and_lookup(&fixture, &committed, |_| None)
                .expect("fallback model should resolve");
        assert_eq!(resolved, Some(committed));
    }

    #[test]
    fn fixture_model_missing_default_is_actionable() {
        let fixture = ValidateFixture {
            model_env: Some("RAVE_VALIDATE_MODEL".into()),
            ..ValidateFixture::default()
        };
        let missing = std::env::temp_dir().join(format!(
            "rave_missing_model_{}_{}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let err = resolve_fixture_model_path_with_default_and_lookup(&fixture, &missing, |_| None)
            .expect_err("missing fallback model should fail");
        let msg = err.to_string();
        assert!(msg.contains("default model not found"));
        assert!(msg.contains("RAVE_VALIDATE_MODEL"));
        assert!(msg.contains(missing.to_string_lossy().as_ref()));
    }
}
