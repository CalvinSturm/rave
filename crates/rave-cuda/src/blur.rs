//! GPU face-blur kernels and region compositing helpers.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};

use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::types::{GpuTexture, PixelFormat};

const MODULE_NAME: &str = "rave_blur";
const KERNEL_NAMES: &[&str] = &[
    "box_blur_h_rgb_planar_f32",
    "box_blur_v_rgb_planar_f32",
    "composite_regions_rgb_planar_f32",
];

const BLUR_CUDA_SRC: &str = r#"
extern "C" __global__ void box_blur_h_rgb_planar_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int radius,
    int channel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        int count = 0;
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = x + dx;
            if (xx < 0) xx = 0;
            if (xx >= width) xx = width - 1;
            int in_idx = c * channel_stride + y * width + xx;
            sum += input[in_idx];
            count += 1;
        }
        output[c * channel_stride + idx] = sum / (float)count;
    }
}

extern "C" __global__ void box_blur_v_rgb_planar_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int radius,
    int channel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        int count = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            int yy = y + dy;
            if (yy < 0) yy = 0;
            if (yy >= height) yy = height - 1;
            int in_idx = c * channel_stride + yy * width + x;
            sum += input[in_idx];
            count += 1;
        }
        output[c * channel_stride + idx] = sum / (float)count;
    }
}

extern "C" __global__ void composite_regions_rgb_planar_f32(
    const float* __restrict__ original,
    const float* __restrict__ blurred,
    float* __restrict__ output,
    int width,
    int height,
    int channel_stride,
    const unsigned int* __restrict__ rects,
    int rect_count)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    bool use_blur = (rect_count == 0);
    for (int i = 0; i < rect_count && !use_blur; ++i) {
        unsigned int rx = rects[i * 4 + 0];
        unsigned int ry = rects[i * 4 + 1];
        unsigned int rw = rects[i * 4 + 2];
        unsigned int rh = rects[i * 4 + 3];
        if ((unsigned int)x >= rx && (unsigned int)x < (rx + rw) &&
            (unsigned int)y >= ry && (unsigned int)y < (ry + rh)) {
            use_blur = true;
        }
    }

    int idx = y * width + x;
    for (int c = 0; c < 3; ++c) {
        int off = c * channel_stride + idx;
        output[off] = use_blur ? blurred[off] : original[off];
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlurRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub struct FaceBlurConfig {
    pub sigma: f32,
    pub iterations: u32,
    pub regions: Option<Vec<BlurRegion>>,
}

impl FaceBlurConfig {
    pub fn validate(&self) -> Result<()> {
        if self.iterations == 0 {
            return Err(EngineError::InvariantViolation(
                "FaceBlur config requires iterations > 0".into(),
            ));
        }
        if !(self.sigma.is_finite() && self.sigma > 0.0) {
            return Err(EngineError::InvariantViolation(
                "FaceBlur config requires sigma > 0".into(),
            ));
        }
        Ok(())
    }

    fn radius(&self) -> i32 {
        self.sigma.ceil().max(1.0) as i32
    }
}

struct BlurScratch {
    width: u32,
    height: u32,
    tmp_a: Arc<cudarc::driver::CudaSlice<u8>>,
    tmp_b: Arc<cudarc::driver::CudaSlice<u8>>,
    region_key: Vec<BlurRegion>,
    region_data: Option<cudarc::driver::CudaSlice<u32>>,
}

pub struct FaceBlurEngine {
    box_blur_h: CudaFunction,
    box_blur_v: CudaFunction,
    composite: CudaFunction,
    scratch: Option<BlurScratch>,
}

impl FaceBlurEngine {
    pub fn compile(device: &Arc<CudaDevice>) -> Result<Self> {
        let ptx = cudarc::nvrtc::compile_ptx(BLUR_CUDA_SRC)?;
        device.load_ptx(ptx, MODULE_NAME, KERNEL_NAMES)?;

        let get_fn = |name: &str| -> Result<CudaFunction> {
            device.get_func(MODULE_NAME, name).ok_or_else(|| {
                EngineError::ModelMetadata(format!(
                    "Blur kernel function '{name}' not found in module '{MODULE_NAME}'"
                ))
            })
        };

        Ok(Self {
            box_blur_h: get_fn("box_blur_h_rgb_planar_f32")?,
            box_blur_v: get_fn("box_blur_v_rgb_planar_f32")?,
            composite: get_fn("composite_regions_rgb_planar_f32")?,
            scratch: None,
        })
    }

    pub fn blur_rgb_planar_f32(
        &mut self,
        input: &GpuTexture,
        config: &FaceBlurConfig,
        ctx: &GpuContext,
        stream: &CudaStream,
    ) -> Result<GpuTexture> {
        config.validate()?;
        if input.format != PixelFormat::RgbPlanarF32 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::RgbPlanarF32,
                actual: input.format,
            });
        }

        self.ensure_scratch(input.width, input.height, config, ctx)?;
        let scratch = self.scratch.as_mut().expect("scratch must be initialized");

        let cfg = launch_config_2d(input.width, input.height);
        let channel_stride = (input.width as i32) * (input.height as i32);

        let mut src_ptr = input.device_ptr();
        for iter in 0..config.iterations {
            let horiz_ptr = if iter % 2 == 0 {
                *scratch.tmp_a.device_ptr()
            } else {
                *scratch.tmp_b.device_ptr()
            };
            let vert_ptr = if iter % 2 == 0 {
                *scratch.tmp_b.device_ptr()
            } else {
                *scratch.tmp_a.device_ptr()
            };
            // SAFETY: pointers are valid GPU buffers with matching shape.
            unsafe {
                self.box_blur_h.clone().launch_on_stream(
                    stream,
                    cfg,
                    (
                        src_ptr,
                        horiz_ptr,
                        input.width as i32,
                        input.height as i32,
                        config.radius(),
                        channel_stride,
                    ),
                )?;
                self.box_blur_v.clone().launch_on_stream(
                    stream,
                    cfg,
                    (
                        horiz_ptr,
                        vert_ptr,
                        input.width as i32,
                        input.height as i32,
                        config.radius(),
                        channel_stride,
                    ),
                )?;
            }
            src_ptr = vert_ptr;
        }

        let out_bytes = input.byte_size();
        let out_buf = ctx.alloc(out_bytes)?;
        let out_ptr = *out_buf.device_ptr();
        let rect_count = scratch
            .region_data
            .as_ref()
            .map(|_| scratch.region_key.len() as i32)
            .unwrap_or(0);
        let rect_ptr = scratch
            .region_data
            .as_ref()
            .map(|v| *v.device_ptr())
            .unwrap_or(0);

        // SAFETY: pointers are valid and region metadata count matches region buffer.
        unsafe {
            self.composite.clone().launch_on_stream(
                stream,
                cfg,
                (
                    input.device_ptr(),
                    src_ptr,
                    out_ptr,
                    input.width as i32,
                    input.height as i32,
                    channel_stride,
                    rect_ptr,
                    rect_count,
                ),
            )?;
        }

        Ok(GpuTexture {
            data: Arc::new(out_buf),
            width: input.width,
            height: input.height,
            pitch: (input.width as usize) * 4,
            format: PixelFormat::RgbPlanarF32,
        })
    }

    fn ensure_scratch(
        &mut self,
        width: u32,
        height: u32,
        config: &FaceBlurConfig,
        ctx: &GpuContext,
    ) -> Result<()> {
        let need_scratch = match &self.scratch {
            Some(s) => s.width != width || s.height != height,
            None => true,
        };

        if need_scratch {
            let bytes = PixelFormat::RgbPlanarF32.byte_size(width, height, (width as usize) * 4);
            let tmp_a = Arc::new(ctx.alloc(bytes)?);
            let tmp_b = Arc::new(ctx.alloc(bytes)?);
            self.scratch = Some(BlurScratch {
                width,
                height,
                tmp_a,
                tmp_b,
                region_key: Vec::new(),
                region_data: None,
            });
        }

        let scratch = self.scratch.as_mut().expect("scratch must be initialized");
        let regions = config.regions.clone().unwrap_or_default();
        if scratch.region_key != regions {
            scratch.region_key = regions.clone();
            if regions.is_empty() {
                scratch.region_data = None;
            } else {
                let flat = flatten_regions(&regions);
                scratch.region_data = Some(ctx.device().htod_sync_copy(&flat)?);
            }
        }

        Ok(())
    }
}

fn launch_config_2d(width: u32, height: u32) -> LaunchConfig {
    let block = (16u32, 16u32, 1u32);
    let grid = (width.div_ceil(block.0), height.div_ceil(block.1), 1u32);
    LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    }
}

fn flatten_regions(regions: &[BlurRegion]) -> Vec<u32> {
    let mut out = Vec::with_capacity(regions.len() * 4);
    for rect in regions {
        out.push(rect.x);
        out.push(rect.y);
        out.push(rect.width);
        out.push(rect.height);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{BlurRegion, FaceBlurConfig};

    fn ref_blur(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let channel_stride = width * height;
        let mut out = vec![0.0f32; input.len()];
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for dy in -(radius as isize)..=(radius as isize) {
                        let yy = (y as isize + dy).clamp(0, (height - 1) as isize) as usize;
                        for dx in -(radius as isize)..=(radius as isize) {
                            let xx = (x as isize + dx).clamp(0, (width - 1) as isize) as usize;
                            sum += input[c * channel_stride + yy * width + xx];
                            count += 1;
                        }
                    }
                    out[c * channel_stride + y * width + x] = sum / count as f32;
                }
            }
        }
        out
    }

    fn ref_composite(
        original: &[f32],
        blurred: &[f32],
        width: usize,
        height: usize,
        regions: &[BlurRegion],
    ) -> Vec<f32> {
        let channel_stride = width * height;
        let mut out = vec![0.0f32; original.len()];
        for y in 0..height {
            for x in 0..width {
                let in_region = regions.is_empty()
                    || regions.iter().any(|r| {
                        x >= r.x as usize
                            && x < (r.x + r.width) as usize
                            && y >= r.y as usize
                            && y < (r.y + r.height) as usize
                    });
                for c in 0..3 {
                    let idx = c * channel_stride + y * width + x;
                    out[idx] = if in_region {
                        blurred[idx]
                    } else {
                        original[idx]
                    };
                }
            }
        }
        out
    }

    #[test]
    fn config_validation_rejects_bad_values() {
        let bad_sigma = FaceBlurConfig {
            sigma: 0.0,
            iterations: 1,
            regions: None,
        };
        assert!(bad_sigma.validate().is_err());

        let bad_iter = FaceBlurConfig {
            sigma: 1.0,
            iterations: 0,
            regions: None,
        };
        assert!(bad_iter.validate().is_err());
    }

    #[test]
    fn reference_region_composite_only_blurs_selected_area() {
        let width = 4usize;
        let height = 4usize;
        let channel_stride = width * height;
        let mut input = vec![0.0f32; 3 * channel_stride];
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let base = (x * x + y * 3 + c) as f32;
                    input[c * channel_stride + y * width + x] = base;
                }
            }
        }

        let blurred = ref_blur(&input, width, height, 1);
        let region = [BlurRegion {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        }];
        let out = ref_composite(&input, &blurred, width, height, &region);

        let outside_idx = 0usize;
        assert_eq!(out[outside_idx], input[outside_idx]);

        let inside_xy = 2 + 2 * width;
        assert_ne!(out[inside_xy], input[inside_xy]);
    }

    #[test]
    fn reference_blur_is_deterministic() {
        let width = 3usize;
        let height = 3usize;
        let data = vec![0.25f32; 3 * width * height];
        let a = ref_blur(&data, width, height, 1);
        let b = ref_blur(&data, width, height, 1);
        assert_eq!(a, b);
    }
}
