//! CUDA preprocessing kernels — NV12↔RGB conversion, FP16, and normalization.
//!
//! All transforms execute on-device via NVRTC-compiled kernels.
//! No host staging.  No CPU-side pixel manipulation.
//!
//! # Kernel compilation
//!
//! CUDA C source is compiled to PTX **once** at engine startup via NVRTC.
//! The PTX is loaded into the `CudaDevice` as a named module.
//! Kernel function handles are resolved once and reused for all frames.
//!
//! # Color space
//!
//! NV12↔RGB uses **BT.709** coefficients (the standard for HD/4K content).
//! The conversion is not configurable at runtime — changing color matrices
//! requires recompiling the kernel source.
//!
//! # Transform chain
//!
//! ```text
//! Decode (NV12)
//!   ├── nv12_to_rgb_planar_f32  → RgbPlanarF32  (for F32 models)
//!   │       └── f32_to_f16      → RgbPlanarF16  (if model wants F16)
//!   └── nv12_to_rgb_planar_f16  → RgbPlanarF16  (fused, skip F32)
//!
//! Inference output (RgbPlanarF32 or RgbPlanarF16)
//!   ├── f16_to_f32              → RgbPlanarF32  (if model outputs F16)
//!   └── rgb_planar_f32_to_nv12  → NV12          → Encode
//! ```
//!
//! # Batch dimension
//!
//! "Batch injection" is a metadata annotation — the underlying buffer is
//! already `1×C×H×W` in memory.  [`ModelInput::from_texture`] annotates the
//! batch dimension without any data copy.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use tracing::info;

use crate::sys::{self, CUevent};
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::types::{GpuTexture, PixelFormat};

// ─── CUDA C kernel source ────────────────────────────────────────────────────

/// CUDA C source for all preprocessing kernels.
///
/// Compiled to PTX once via NVRTC at engine initialization.
const PREPROCESS_CUDA_SRC: &str = r#"
#include <cuda_fp16.h>

// ============================================================================
// BT.709 NV12 → RGB Planar Float32 (NCHW, [0,1])
// ============================================================================
extern "C" __global__ void nv12_to_rgb_planar_f32(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ uv_plane,
    float*               __restrict__ output,
    int width,
    int height,
    int y_pitch,
    int uv_pitch,
    int channel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float Y = (float)y_plane[y * y_pitch + x];

    // UV is sub-sampled 2x2, interleaved [U V U V ...]
    int uv_x = x >> 1;
    int uv_y = y >> 1;
    float U = (float)uv_plane[uv_y * uv_pitch + uv_x * 2    ];
    float V = (float)uv_plane[uv_y * uv_pitch + uv_x * 2 + 1];

    // BT.709 full-range conversion
    float Yn = Y / 255.0f;
    float Un = (U - 128.0f) / 255.0f;
    float Vn = (V - 128.0f) / 255.0f;

    float r = Yn + 1.5748f * Vn;
    float g = Yn - 0.1873f * Un - 0.4681f * Vn;
    float b = Yn + 1.8556f * Un;

    // Clamp to [0, 1]
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    // Write NCHW planar: [R plane][G plane][B plane]
    int idx = y * width + x;
    output[0 * channel_stride + idx] = r;
    output[1 * channel_stride + idx] = g;
    output[2 * channel_stride + idx] = b;
}

// ============================================================================
// BT.709 NV12 → RGB Planar Float16 (NCHW, [0,1]) — FUSED, no F32 intermediate
// ============================================================================
extern "C" __global__ void nv12_to_rgb_planar_f16(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ uv_plane,
    __half*              __restrict__ output,
    int width,
    int height,
    int y_pitch,
    int uv_pitch,
    int channel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float Y = (float)y_plane[y * y_pitch + x];

    int uv_x = x >> 1;
    int uv_y = y >> 1;
    float U = (float)uv_plane[uv_y * uv_pitch + uv_x * 2    ];
    float V = (float)uv_plane[uv_y * uv_pitch + uv_x * 2 + 1];

    float Yn = Y / 255.0f;
    float Un = (U - 128.0f) / 255.0f;
    float Vn = (V - 128.0f) / 255.0f;

    float r = fminf(fmaxf(Yn + 1.5748f * Vn, 0.0f), 1.0f);
    float g = fminf(fmaxf(Yn - 0.1873f * Un - 0.4681f * Vn, 0.0f), 1.0f);
    float b = fminf(fmaxf(Yn + 1.8556f * Un, 0.0f), 1.0f);

    int idx = y * width + x;
    output[0 * channel_stride + idx] = __float2half(r);
    output[1 * channel_stride + idx] = __float2half(g);
    output[2 * channel_stride + idx] = __float2half(b);
}

// ============================================================================
// RGB Planar Float32 → Float16 (element-wise conversion, 1D grid)
// ============================================================================
extern "C" __global__ void f32_to_f16(
    const float* __restrict__ input,
    __half*      __restrict__ output,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = __float2half(input[i]);
}

// ============================================================================
// RGB Planar Float16 → Float32 (element-wise conversion, 1D grid)
// ============================================================================
extern "C" __global__ void f16_to_f32(
    const __half* __restrict__ input,
    float*        __restrict__ output,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = __half2float(input[i]);
}

// ============================================================================
// RGB Planar Float32 (NCHW, [0,1]) → NV12
// ============================================================================
extern "C" __global__ void rgb_planar_f32_to_nv12(
    const float*         __restrict__ input,
    unsigned char*       __restrict__ y_plane,
    unsigned char*       __restrict__ uv_plane,
    int width,
    int height,
    int y_pitch,
    int uv_pitch,
    int channel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float r = input[0 * channel_stride + idx];
    float g = input[1 * channel_stride + idx];
    float b = input[2 * channel_stride + idx];

    // BT.709 forward: RGB [0,1] → YUV
    float Y_val = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float U_val = -0.1146f * r - 0.3854f * g + 0.5f    * b;
    float V_val = 0.5f    * r - 0.4542f * g - 0.0458f * b;

    // Scale to [0,255] for Y, [0,255] centered at 128 for UV
    y_plane[y * y_pitch + x] = (unsigned char)fminf(fmaxf(Y_val * 255.0f + 0.5f, 0.0f), 255.0f);

    // UV: only write for even (x,y) positions to produce sub-sampled plane
    if ((x & 1) == 0 && (y & 1) == 0) {
        // Average the 2x2 block for chroma sub-sampling
        float u_sum = U_val;
        float v_sum = V_val;
        int count = 1;

        if (x + 1 < width) {
            int idx2 = y * width + (x + 1);
            float r2 = input[0 * channel_stride + idx2];
            float g2 = input[1 * channel_stride + idx2];
            float b2 = input[2 * channel_stride + idx2];
            u_sum += -0.1146f * r2 - 0.3854f * g2 + 0.5f * b2;
            v_sum += 0.5f * r2 - 0.4542f * g2 - 0.0458f * b2;
            count++;
        }
        if (y + 1 < height) {
            int idx3 = (y + 1) * width + x;
            float r3 = input[0 * channel_stride + idx3];
            float g3 = input[1 * channel_stride + idx3];
            float b3 = input[2 * channel_stride + idx3];
            u_sum += -0.1146f * r3 - 0.3854f * g3 + 0.5f * b3;
            v_sum += 0.5f * r3 - 0.4542f * g3 - 0.0458f * b3;
            count++;
        }
        if (x + 1 < width && y + 1 < height) {
            int idx4 = (y + 1) * width + (x + 1);
            float r4 = input[0 * channel_stride + idx4];
            float g4 = input[1 * channel_stride + idx4];
            float b4 = input[2 * channel_stride + idx4];
            u_sum += -0.1146f * r4 - 0.3854f * g4 + 0.5f * b4;
            v_sum += 0.5f * r4 - 0.4542f * g4 - 0.0458f * b4;
            count++;
        }

        float u_avg = u_sum / (float)count;
        float v_avg = v_sum / (float)count;

        int uv_idx = (y >> 1) * uv_pitch + (x >> 1) * 2;
        uv_plane[uv_idx    ] = (unsigned char)fminf(fmaxf(u_avg * 255.0f + 128.0f + 0.5f, 0.0f), 255.0f);
        uv_plane[uv_idx + 1] = (unsigned char)fminf(fmaxf(v_avg * 255.0f + 128.0f + 0.5f, 0.0f), 255.0f);
    }
}
"#;

const MODULE_NAME: &str = "rave_preprocess";

/// All NVRTC function names — compiled and resolved once.
const KERNEL_NAMES: &[&str] = &[
    "nv12_to_rgb_planar_f32",
    "nv12_to_rgb_planar_f16",
    "f32_to_f16",
    "f16_to_f32",
    "rgb_planar_f32_to_nv12",
];

// ─── Compiled kernel handles ─────────────────────────────────────────────────

/// Holds resolved function handles for all preprocessing kernels.
///
/// Created once during engine initialization, reused for every frame.
/// **No per-frame PTX recompilation.**
pub struct PreprocessKernels {
    _device: Arc<CudaDevice>,
    nv12_to_rgb_f32: CudaFunction,
    nv12_to_rgb_f16: CudaFunction,
    f32_to_f16: CudaFunction,
    f16_to_f32: CudaFunction,
    rgb_f32_to_nv12: CudaFunction,
}

impl PreprocessKernels {
    /// Compile the CUDA C source via NVRTC and load kernel handles.
    ///
    /// Call once at engine startup.  All subsequent launches reuse the
    /// compiled module — zero recompilation cost per frame.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::NvrtcCompile`] if compilation fails (driver mismatch).
    /// Returns [`EngineError::Cuda`] if module loading fails.
    pub fn compile(device: &Arc<CudaDevice>) -> Result<Self> {
        let mut extra_nvrtc_options = Vec::new();
        let resolved_include_dir = resolve_cuda_include_dir();

        if let Some(include_dir) = resolved_include_dir.as_ref() {
            // NVRTC accepts both -I<dir> and --include-path=<dir>.
            // Keep both to improve portability across platform-specific drivers.
            extra_nvrtc_options.push(format!("-I{}", include_dir.display()));
            extra_nvrtc_options.push(format!("--include-path={}", include_dir.display()));
        }

        let mut final_nvrtc_options = vec![
            "--ftz=true".to_string(),
            "--prec-sqrt=false".to_string(),
            "--prec-div=false".to_string(),
        ];
        final_nvrtc_options.extend(extra_nvrtc_options.iter().cloned());
        info!(
            resolved_cuda_include_path = ?resolved_include_dir
                .as_ref()
                .map(|p| p.display().to_string()),
            nvrtc_options = ?final_nvrtc_options,
            "NVRTC compile options resolved"
        );

        // Compile with FP16 support enabled.
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            PREPROCESS_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                ftz: Some(true),        // Flush denorms to zero.
                prec_div: Some(false),  // Fast division.
                prec_sqrt: Some(false), // Fast sqrt.
                options: extra_nvrtc_options,
                ..Default::default()
            },
        )?;

        device.load_ptx(ptx, MODULE_NAME, KERNEL_NAMES)?;

        let get_fn = |name: &str| -> Result<CudaFunction> {
            device.get_func(MODULE_NAME, name).ok_or_else(|| {
                EngineError::ModelMetadata(format!(
                    "Kernel function '{name}' not found in module '{MODULE_NAME}'"
                ))
            })
        };

        let kernels = Self {
            _device: Arc::clone(device),
            nv12_to_rgb_f32: get_fn("nv12_to_rgb_planar_f32")?,
            nv12_to_rgb_f16: get_fn("nv12_to_rgb_planar_f16")?,
            f32_to_f16: get_fn("f32_to_f16")?,
            f16_to_f32: get_fn("f16_to_f32")?,
            rgb_f32_to_nv12: get_fn("rgb_planar_f32_to_nv12")?,
        };

        info!(
            "NVRTC: compiled {} kernels from module '{MODULE_NAME}'",
            KERNEL_NAMES.len()
        );
        Ok(kernels)
    }

    // ── NV12 → RGB F32 ──────────────────────────────────────────────────

    /// Convert an NV12 `GpuTexture` to `RgbPlanarF32` on the given stream.
    ///
    /// Output is NCHW planar, `[0,1]` normalized, BT.709.
    /// Allocates the output buffer via `ctx` (pooled when possible).
    ///
    /// # Invariants
    ///
    /// - `input.format` must be `PixelFormat::Nv12`.
    /// - `stream` must belong to the same CUDA context as `input.data`.
    /// - The output buffer must not be read until the stream is synchronized
    ///   (or a downstream `cuStreamWaitEvent` ensures ordering).
    pub fn nv12_to_rgb(
        &self,
        input: &GpuTexture,
        ctx: &GpuContext,
        stream: &CudaStream,
    ) -> Result<GpuTexture> {
        if input.format != PixelFormat::Nv12 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::Nv12,
                actual: input.format,
            });
        }

        let w = input.width as usize;
        let h = input.height as usize;
        let channel_stride = w * h;
        let out_bytes = 3 * channel_stride * std::mem::size_of::<f32>();

        let output_buf = ctx.alloc(out_bytes)?;

        let y_ptr = input.device_ptr();
        let uv_ptr = y_ptr + (input.pitch * h) as u64;
        let out_ptr = *output_buf.device_ptr();

        let (config, _block) = launch_config_2d(input.width, input.height);

        // SAFETY:
        // - All pointers are valid device pointers from the same CudaDevice.
        // - `input.data` remains alive (held by the caller's GpuTexture).
        // - `output_buf` remains alive until wrapped in the returned GpuTexture.
        // - Grid/block dimensions cover [0..width) × [0..height).
        // - Kernel reads Y at [y_ptr, y_ptr + pitch*height) and UV at
        //   [uv_ptr, uv_ptr + pitch*(height/2)), both within `input.data`.
        // - Kernel writes [out_ptr, out_ptr + out_bytes), within `output_buf`.
        unsafe {
            self.nv12_to_rgb_f32.clone().launch_on_stream(
                stream,
                config,
                (
                    y_ptr,
                    uv_ptr,
                    out_ptr,
                    input.width as i32,
                    input.height as i32,
                    input.pitch as i32,
                    input.pitch as i32, // NV12: UV pitch == Y pitch
                    channel_stride as i32,
                ),
            )?;
        }

        Ok(GpuTexture {
            data: Arc::new(output_buf),
            width: input.width,
            height: input.height,
            pitch: w * std::mem::size_of::<f32>(), // dense planar pitch
            format: PixelFormat::RgbPlanarF32,
        })
    }

    // ── NV12 → RGB F16 (fused) ──────────────────────────────────────────

    /// Convert NV12 directly to `RgbPlanarF16` — skips F32 intermediate.
    ///
    /// Use when the inference model expects FP16 input.  Saves one
    /// kernel launch and one buffer allocation vs. `nv12_to_rgb` + `f32_to_f16`.
    pub fn nv12_to_rgb_f16(
        &self,
        input: &GpuTexture,
        ctx: &GpuContext,
        stream: &CudaStream,
    ) -> Result<GpuTexture> {
        if input.format != PixelFormat::Nv12 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::Nv12,
                actual: input.format,
            });
        }

        let w = input.width as usize;
        let h = input.height as usize;
        let channel_stride = w * h;
        let out_bytes = PixelFormat::RgbPlanarF16.byte_size(input.width, input.height, 0);

        let output_buf = ctx.alloc(out_bytes)?;

        let y_ptr = input.device_ptr();
        let uv_ptr = y_ptr + (input.pitch * h) as u64;
        let out_ptr = *output_buf.device_ptr();

        let (config, _) = launch_config_2d(input.width, input.height);

        // SAFETY: same invariants as nv12_to_rgb, output is F16.
        unsafe {
            self.nv12_to_rgb_f16.clone().launch_on_stream(
                stream,
                config,
                (
                    y_ptr,
                    uv_ptr,
                    out_ptr,
                    input.width as i32,
                    input.height as i32,
                    input.pitch as i32,
                    input.pitch as i32,
                    channel_stride as i32,
                ),
            )?;
        }

        Ok(GpuTexture {
            data: Arc::new(output_buf),
            width: input.width,
            height: input.height,
            pitch: w * 2, // F16 element = 2 bytes
            format: PixelFormat::RgbPlanarF16,
        })
    }

    // ── F32 → F16 ────────────────────────────────────────────────────────

    /// Convert `RgbPlanarF32` to `RgbPlanarF16` (element-wise truncation).
    ///
    /// 1D kernel launch — flat array, no layout change.
    pub fn convert_f32_to_f16(
        &self,
        input: &GpuTexture,
        ctx: &GpuContext,
        stream: &CudaStream,
    ) -> Result<GpuTexture> {
        if input.format != PixelFormat::RgbPlanarF32 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::RgbPlanarF32,
                actual: input.format,
            });
        }

        let count = 3 * (input.width as usize) * (input.height as usize);
        let out_bytes = count * 2; // FP16 = 2 bytes

        let output_buf = ctx.alloc(out_bytes)?;

        let in_ptr = input.device_ptr();
        let out_ptr = *output_buf.device_ptr();

        let config = launch_config_1d(count);

        // SAFETY: in_ptr has `count` f32 elements, out_ptr has `count` f16 slots.
        unsafe {
            self.f32_to_f16.clone().launch_on_stream(
                stream,
                config,
                (in_ptr, out_ptr, count as i32),
            )?;
        }

        Ok(GpuTexture {
            data: Arc::new(output_buf),
            width: input.width,
            height: input.height,
            pitch: (input.width as usize) * 2,
            format: PixelFormat::RgbPlanarF16,
        })
    }

    // ── F16 → F32 ────────────────────────────────────────────────────────

    /// Convert `RgbPlanarF16` to `RgbPlanarF32` (element-wise promotion).
    ///
    /// Used to convert FP16 model output back to F32 for RGB→NV12 conversion.
    pub fn convert_f16_to_f32(
        &self,
        input: &GpuTexture,
        ctx: &GpuContext,
        stream: &CudaStream,
    ) -> Result<GpuTexture> {
        if input.format != PixelFormat::RgbPlanarF16 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::RgbPlanarF16,
                actual: input.format,
            });
        }

        let count = 3 * (input.width as usize) * (input.height as usize);
        let out_bytes = count * 4; // F32 = 4 bytes

        let output_buf = ctx.alloc(out_bytes)?;

        let in_ptr = input.device_ptr();
        let out_ptr = *output_buf.device_ptr();

        let config = launch_config_1d(count);

        // SAFETY: in_ptr has `count` f16 elements, out_ptr has `count` f32 slots.
        unsafe {
            self.f16_to_f32.clone().launch_on_stream(
                stream,
                config,
                (in_ptr, out_ptr, count as i32),
            )?;
        }

        Ok(GpuTexture {
            data: Arc::new(output_buf),
            width: input.width,
            height: input.height,
            pitch: (input.width as usize) * 4,
            format: PixelFormat::RgbPlanarF32,
        })
    }

    // ── RGB F32 → NV12 ──────────────────────────────────────────────────

    /// Convert `RgbPlanarF32` back to NV12 for NVENC input.
    ///
    /// Used after inference to prepare the output for hardware encoding.
    pub fn rgb_to_nv12(
        &self,
        input: &GpuTexture,
        nv12_pitch: usize,
        ctx: &GpuContext,
        stream: &CudaStream,
    ) -> Result<GpuTexture> {
        if input.format != PixelFormat::RgbPlanarF32 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::RgbPlanarF32,
                actual: input.format,
            });
        }

        let w = input.width as usize;
        let h = input.height as usize;
        let channel_stride = w * h;
        let out_bytes = PixelFormat::Nv12.byte_size(input.width, input.height, nv12_pitch);

        let output_buf = ctx.alloc(out_bytes)?;

        let in_ptr = input.device_ptr();
        let y_ptr = *output_buf.device_ptr();
        let uv_ptr = y_ptr + (nv12_pitch * h) as u64;

        let (config, _) = launch_config_2d(input.width, input.height);

        // SAFETY:
        // - `in_ptr` points to valid RgbPlanarF32 device memory (3 × W×H × f32).
        // - `output_buf` has capacity for NV12 at the specified pitch.
        // - Grid/block covers [0..width) × [0..height).
        // - Chroma sub-sampling writes only at even (x,y) positions.
        unsafe {
            self.rgb_f32_to_nv12.clone().launch_on_stream(
                stream,
                config,
                (
                    in_ptr,
                    y_ptr,
                    uv_ptr,
                    input.width as i32,
                    input.height as i32,
                    nv12_pitch as i32,
                    nv12_pitch as i32,
                    channel_stride as i32,
                ),
            )?;
        }

        Ok(GpuTexture {
            data: Arc::new(output_buf),
            width: input.width,
            height: input.height,
            pitch: nv12_pitch,
            format: PixelFormat::Nv12,
        })
    }
}

fn resolve_cuda_include_dir() -> Option<PathBuf> {
    let cuda_path = std::env::var("CUDA_PATH").ok();
    let cuda_home = std::env::var("CUDA_HOME").ok();
    resolve_cuda_include_dir_with(cuda_path.as_deref(), cuda_home.as_deref(), |candidate| {
        candidate.is_dir()
    })
}

fn resolve_cuda_include_dir_with<F>(
    cuda_path: Option<&str>,
    cuda_home: Option<&str>,
    exists: F,
) -> Option<PathBuf>
where
    F: Fn(&Path) -> bool,
{
    let mut candidates = Vec::new();
    if let Some(path) = cuda_path {
        candidates.push(PathBuf::from(path).join("include"));
    }
    if let Some(path) = cuda_home {
        candidates.push(PathBuf::from(path).join("include"));
    }
    #[cfg(unix)]
    {
        candidates.push(PathBuf::from("/usr/local/cuda/include"));
    }
    #[cfg(windows)]
    {
        if let Ok(entries) =
            std::fs::read_dir(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        {
            let mut include_dirs = entries
                .flatten()
                .map(|e| e.path().join("include"))
                .collect::<Vec<_>>();
            include_dirs.sort();
            include_dirs.reverse();
            candidates.extend(include_dirs);
        }
    }

    candidates.into_iter().find(|candidate| exists(candidate))
}

#[cfg(test)]
mod tests {
    use super::resolve_cuda_include_dir_with;

    #[test]
    fn prefers_cuda_path_include_when_present() {
        let resolved = resolve_cuda_include_dir_with(
            Some("/opt/cuda-from-path"),
            Some("/opt/cuda-from-home"),
            |p| p == std::path::Path::new("/opt/cuda-from-path/include"),
        );
        assert_eq!(
            resolved.as_deref(),
            Some(std::path::Path::new("/opt/cuda-from-path/include"))
        );
    }

    #[test]
    fn falls_back_to_cuda_home_include_when_cuda_path_missing() {
        let resolved = resolve_cuda_include_dir_with(
            Some("/opt/cuda-from-path"),
            Some("/opt/cuda-from-home"),
            |p| p == std::path::Path::new("/opt/cuda-from-home/include"),
        );
        assert_eq!(
            resolved.as_deref(),
            Some(std::path::Path::new("/opt/cuda-from-home/include"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn falls_back_to_usr_local_cuda_include() {
        let resolved =
            resolve_cuda_include_dir_with(Some("/missing/path"), Some("/missing/home"), |p| {
                p == std::path::Path::new("/usr/local/cuda/include")
            });
        assert_eq!(
            resolved.as_deref(),
            Some(std::path::Path::new("/usr/local/cuda/include"))
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  BATCH DIMENSION INJECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Model-ready tensor descriptor — annotates batch dimension without copying.
///
/// The underlying `GpuTexture` data is already laid out as `C×H×W` in memory.
/// This struct adds the `N=1` batch dimension as metadata, producing
/// a `1×C×H×W` view for the ORT I/O binding.
///
/// **Zero data copy.**  The device pointer and byte count are unchanged.
#[derive(Debug)]
pub struct ModelInput {
    /// The underlying GPU texture (holds the device memory reference).
    pub texture: GpuTexture,
    /// Tensor shape as `[N, C, H, W]`.
    pub shape: [usize; 4],
    /// Element count: `N × C × H × W`.
    pub element_count: usize,
}

impl ModelInput {
    /// Create a model input from a planar RGB texture.
    ///
    /// Injects `N=1` batch dimension — no data copy.
    ///
    /// # Requirements
    ///
    /// `texture.format` must be `RgbPlanarF32` or `RgbPlanarF16`.
    pub fn from_texture(texture: GpuTexture) -> Result<Self> {
        let channels = match texture.format {
            PixelFormat::RgbPlanarF32 | PixelFormat::RgbPlanarF16 => 3,
            other => {
                return Err(EngineError::FormatMismatch {
                    expected: PixelFormat::RgbPlanarF32,
                    actual: other,
                });
            }
        };

        let w = texture.width as usize;
        let h = texture.height as usize;
        let element_count = channels * h * w;

        Ok(Self {
            shape: [1, channels, h, w],
            element_count,
            texture,
        })
    }

    /// Device pointer to the tensor data.
    pub fn device_ptr(&self) -> u64 {
        self.texture.device_ptr()
    }

    /// Element size in bytes (2 for F16, 4 for F32).
    pub fn element_bytes(&self) -> usize {
        self.texture.format.element_bytes()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  STAGE METRICS — per-kernel GPU timing via CUDA events
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU-side timing for a single kernel launch.
///
/// Uses `cuEventRecord` before and after the kernel to measure GPU execution
/// time without CPU-blocking synchronization during measurement.  The elapsed
/// time is read lazily after the stream is synchronized at stage boundaries.
pub struct KernelTimer {
    start: CUevent,
    end: CUevent,
}

impl KernelTimer {
    /// Create a new timer pair.
    pub fn new() -> Result<Self> {
        let mut start: CUevent = std::ptr::null_mut();
        let mut end: CUevent = std::ptr::null_mut();
        // NOTE: we do NOT use CU_EVENT_DISABLE_TIMING here — we need timing.
        // SAFETY: pointers refer to local CUevent out parameters.
        let rc_start = unsafe { sys::cu_event_create(&mut start, 0)? };
        // SAFETY: pointers refer to local CUevent out parameters.
        let rc_end = unsafe { sys::cu_event_create(&mut end, 0)? };
        sys::check_cu(rc_start, "cuEventCreate (start)")?;
        sys::check_cu(rc_end, "cuEventCreate (end)")?;
        Ok(Self { start, end })
    }

    /// Record the start event on the given stream.
    pub fn record_start(&self, stream: &CudaStream) -> Result<()> {
        let raw = crate::stream::get_raw_stream(stream);
        // SAFETY: event/stream are valid CUDA handles managed by this pipeline.
        let rc = unsafe { sys::cu_event_record(self.start, raw)? };
        sys::check_cu(rc, "timer record start")
    }

    /// Record the end event on the given stream.
    pub fn record_end(&self, stream: &CudaStream) -> Result<()> {
        let raw = crate::stream::get_raw_stream(stream);
        // SAFETY: event/stream are valid CUDA handles managed by this pipeline.
        let rc = unsafe { sys::cu_event_record(self.end, raw)? };
        sys::check_cu(rc, "timer record end")
    }

    /// Query elapsed time in milliseconds.
    ///
    /// The stream must have been synchronized (or the end event must have
    /// completed) before calling this.
    pub fn elapsed_ms(&self) -> Result<f32> {
        let mut ms: f32 = 0.0;
        // SAFETY: ms points to valid storage; start/end are valid timer events.
        let rc = unsafe { sys::cu_event_elapsed_time(&mut ms, self.start, self.end)? };
        sys::check_cu(rc, "cuEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for KernelTimer {
    fn drop(&mut self) {
        // SAFETY: handles were created by cuEventCreate; best-effort cleanup on drop.
        let _ = unsafe { sys::cu_event_destroy_v2(self.start) };
        // SAFETY: handles were created by cuEventCreate; best-effort cleanup on drop.
        let _ = unsafe { sys::cu_event_destroy_v2(self.end) };
    }
}

/// Accumulated per-stage latency metrics.
#[derive(Debug, Default)]
pub struct StageMetrics {
    /// Total kernel execution time in milliseconds.
    pub total_ms: f64,
    /// Number of kernel launches measured.
    pub launch_count: u64,
}

impl StageMetrics {
    /// Record a new measurement.
    pub fn record(&mut self, elapsed_ms: f32) {
        self.total_ms += elapsed_ms as f64;
        self.launch_count += 1;
    }

    /// Average kernel latency in milliseconds.
    pub fn avg_ms(&self) -> f64 {
        if self.launch_count == 0 {
            0.0
        } else {
            self.total_ms / self.launch_count as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PREPROCESS PIPELINE — full transform chain
// ═══════════════════════════════════════════════════════════════════════════════

/// Which floating-point precision the inference model expects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelPrecision {
    /// 32-bit single-precision float (full precision, higher VRAM usage).
    F32,
    /// 16-bit half-precision float (faster inference, lower VRAM usage).
    F16,
}

/// Full preprocess transform chain: NV12 → model-ready tensor.
///
/// Encapsulates kernel selection, buffer allocation, and stage metrics.
/// All operations run on `preprocess_stream` — no implicit device sync.
pub struct PreprocessPipeline {
    kernels: PreprocessKernels,
    precision: ModelPrecision,
    /// Latency metrics for the NV12 → RGB conversion stage.
    pub metrics_nv12_to_rgb: StageMetrics,
    /// Latency metrics for the F32 → F16 conversion stage (F16 models only).
    pub metrics_f32_to_f16: StageMetrics,
}

impl PreprocessPipeline {
    pub fn new(kernels: PreprocessKernels, precision: ModelPrecision) -> Self {
        Self {
            kernels,
            precision,
            metrics_nv12_to_rgb: StageMetrics::default(),
            metrics_f32_to_f16: StageMetrics::default(),
        }
    }

    /// Transform NV12 decoded frame → model-ready `ModelInput`.
    ///
    /// Selects the optimal kernel path based on model precision:
    /// - F32: `nv12_to_rgb_f32` → `ModelInput::from_texture`
    /// - F16: `nv12_to_rgb_f16` (fused) → `ModelInput::from_texture`
    ///
    /// No `cudaDeviceSynchronize`.  No host transforms.
    pub fn prepare(
        &mut self,
        input: &GpuTexture,
        ctx: &GpuContext,
        stream: &CudaStream,
    ) -> Result<ModelInput> {
        let rgb_texture = match self.precision {
            ModelPrecision::F32 => self.kernels.nv12_to_rgb(input, ctx, stream)?,
            ModelPrecision::F16 => self.kernels.nv12_to_rgb_f16(input, ctx, stream)?,
        };

        ModelInput::from_texture(rgb_texture)
    }

    /// Convert model output back to NV12 for encoding.
    ///
    /// Handles F16→F32 promotion if needed before RGB→NV12 conversion.
    pub fn postprocess(
        &self,
        output: GpuTexture,
        nv12_pitch: usize,
        ctx: &GpuContext,
        stream: &CudaStream,
    ) -> Result<GpuTexture> {
        let f32_texture = match output.format {
            PixelFormat::RgbPlanarF32 => output,
            PixelFormat::RgbPlanarF16 => self.kernels.convert_f16_to_f32(&output, ctx, stream)?,
            other => {
                return Err(EngineError::FormatMismatch {
                    expected: PixelFormat::RgbPlanarF32,
                    actual: other,
                });
            }
        };

        self.kernels
            .rgb_to_nv12(&f32_texture, nv12_pitch, ctx, stream)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Standard 2D launch config: 16×16 blocks.
fn launch_config_2d(width: u32, height: u32) -> (LaunchConfig, (u32, u32, u32)) {
    let block = (16u32, 16u32, 1u32);
    let grid = (width.div_ceil(block.0), height.div_ceil(block.1), 1u32);
    (
        LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        },
        block,
    )
}

/// Standard 1D launch config: 256 threads per block.
fn launch_config_1d(count: usize) -> LaunchConfig {
    let block = 256u32;
    let grid = ((count as u32).div_ceil(block), 1, 1);
    LaunchConfig {
        grid_dim: grid,
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}
