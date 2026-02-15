//! NVENC hardware encoder — zero-copy GPU-resident NV12 input.
//!
//! # Architecture
//!
//! ```text
//! GpuTexture { NV12 } ──(device ptr)──▸ nvEncRegisterResource(CUDA)
//!                                              │
//!                                    nvEncMapInputResource
//!                                              │
//!                                     nvEncEncodePicture
//!                                              │
//!                                    nvEncLockBitstream
//!                                              │
//!                                    BitstreamSink.write_packet()
//!                                              │
//!                                    nvEncUnlockBitstream
//!                                    nvEncUnmapInputResource
//! ```
//!
//! # Zero-copy input
//!
//! The NV12 `GpuTexture` device pointer is registered directly as an NVENC
//! input resource via `nvEncRegisterResource(CUDADEVICEPTR)`.  No host
//! staging, no intermediate copies (the NVENC ASIC reads from device memory).
//!
//! # Resource registration strategy
//!
//! Each unique device pointer must be registered before use.  Since the
//! pipeline reuses a bounded set of buffers (OutputRing or recycled pool),
//! we cache registrations keyed by device pointer.  A registration is valid
//! as long as the device pointer remains valid (guaranteed by `Arc<CudaSlice>`
//! lifetime in the `FrameEnvelope`).

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use tracing::{debug, info};

use crate::codecs::sys::*;
use crate::core::types::{FrameEnvelope, PixelFormat};
use crate::engine::pipeline::FrameEncoder;
use crate::error::{EngineError, Result};

// ─── Bitstream sink trait ────────────────────────────────────────────────

/// Receives encoded bitstream output.
///
/// Implementations: file writer, muxer, network sender, etc.
pub trait BitstreamSink: Send + 'static {
    fn write_packet(&mut self, data: &[u8], pts: i64, is_keyframe: bool) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
}

// ─── NVENC configuration ─────────────────────────────────────────────────

/// Encoder configuration parameters.
#[derive(Clone, Debug)]
pub struct NvEncConfig {
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
    /// Framerate numerator.
    pub fps_num: u32,
    /// Framerate denominator.
    pub fps_den: u32,
    /// Average bitrate in bits/sec (0 = CQP mode).
    pub bitrate: u32,
    /// Max bitrate in bits/sec (VBR mode).
    pub max_bitrate: u32,
    /// GOP length (frames between IDR).
    pub gop_length: u32,
    /// B-frame interval (0 = no B-frames).
    pub b_frames: u32,
    /// NV12 row pitch (must match incoming frame pitch).
    pub nv12_pitch: u32,
}

// ─── Registration cache ──────────────────────────────────────────────────

/// Caches NVENC resource registrations keyed by device pointer.
///
/// NVENC requires `nvEncRegisterResource` before a device pointer can be
/// used as input.  Since the pipeline reuses a bounded set of buffers,
/// caching registrations avoids per-frame registration overhead.
struct RegistrationCache {
    /// Map from device pointer → registered resource handle.
    entries: HashMap<u64, *mut c_void>,
}

impl RegistrationCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get(&self, dev_ptr: u64) -> Option<*mut c_void> {
        self.entries.get(&dev_ptr).copied()
    }

    fn insert(&mut self, dev_ptr: u64, handle: *mut c_void) {
        self.entries.insert(dev_ptr, handle);
    }

    fn handles(&self) -> impl Iterator<Item = *mut c_void> + '_ {
        self.entries.values().copied()
    }
}

// ─── NvEncoder ───────────────────────────────────────────────────────────

/// NVENC hardware encoder consuming GPU-resident NV12 `FrameEnvelope`s.
///
/// Implements [`FrameEncoder`].
pub struct NvEncoder {
    /// NVENC encoder session handle.
    encoder: *mut c_void,
    /// NVENC function pointer table.
    fns: NV_ENCODE_API_FUNCTION_LIST,
    /// Bitstream output buffer (NVENC-allocated).
    bitstream_buf: *mut c_void,
    /// Output sink for encoded data.
    sink: Box<dyn BitstreamSink>,
    /// Cached resource registrations.
    reg_cache: RegistrationCache,
    /// Encoder configuration.
    config: NvEncConfig,
    /// Frame counter for encode ordering.
    frame_idx: u32,
}

// SAFETY: NvEncoder is only used from the encode stage (single blocking thread).
// The NVENC API is thread-safe for a single session from one thread.
unsafe impl Send for NvEncoder {}

impl NvEncoder {
    /// Create and initialize an NVENC encoder session.
    ///
    /// `cuda_context` is the raw CUcontext handle.  On cudarc, this can
    /// be obtained from the device.
    pub fn new(
        cuda_context: *mut c_void,
        sink: Box<dyn BitstreamSink>,
        config: NvEncConfig,
    ) -> Result<Self> {
        // ── Get function table ──
        let mut fns: NV_ENCODE_API_FUNCTION_LIST = unsafe { std::mem::zeroed() };
        fns.version = nvenc_struct_version(2);

        // SAFETY: fns is zeroed with version set.
        // NvEncodeAPICreateInstance fills the function pointers.
        unsafe {
            check_nvenc(
                NvEncodeAPICreateInstance(&mut fns),
                "NvEncodeAPICreateInstance",
            )?;
        }

        // ── Open session ──
        let mut open_params: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS = unsafe { std::mem::zeroed() };
        open_params.version = nvenc_struct_version(1);
        open_params.deviceType = NV_ENC_DEVICE_TYPE::CUDA;
        open_params.device = cuda_context;
        open_params.apiVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;

        let mut encoder: *mut c_void = ptr::null_mut();
        let open_fn = fns
            .nvEncOpenEncodeSessionEx
            .ok_or_else(|| EngineError::Encode("nvEncOpenEncodeSessionEx not found".into()))?;

        // SAFETY: open_params is fully initialized.
        unsafe {
            check_nvenc(
                open_fn(&mut open_params, &mut encoder),
                "nvEncOpenEncodeSessionEx",
            )?;
        }

        info!(
            width = config.width,
            height = config.height,
            fps = format!("{}/{}", config.fps_num, config.fps_den),
            bitrate = config.bitrate,
            "NVENC session opened"
        );

        // ── Get preset config ──
        let get_preset_fn = fns
            .nvEncGetEncodePresetConfigEx
            .ok_or_else(|| EngineError::Encode("nvEncGetEncodePresetConfigEx not found".into()))?;

        let mut preset_config: NV_ENC_PRESET_CONFIG = unsafe { std::mem::zeroed() };
        preset_config.version = nvenc_struct_version(1);
        preset_config.presetCfg.version = nvenc_struct_version(1);

        unsafe {
            check_nvenc(
                get_preset_fn(
                    encoder,
                    NV_ENC_CODEC_HEVC_GUID,
                    NV_ENC_PRESET_P7_GUID,
                    NV_ENC_TUNING_INFO::HIGH_QUALITY,
                    &mut preset_config,
                ),
                "nvEncGetEncodePresetConfigEx",
            )?;
        }

        // ── Configure encoder ──
        let mut enc_config = preset_config.presetCfg;
        enc_config.version = nvenc_struct_version(1);
        enc_config.profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID;
        enc_config.gopLength = config.gop_length;
        enc_config.frameIntervalP = (config.b_frames + 1) as i32;

        if config.bitrate > 0 {
            // VBR mode.
            enc_config.rcParams.rateControlMode = 2; // NV_ENC_PARAMS_RC_VBR
            enc_config.rcParams.averageBitRate = config.bitrate;
            enc_config.rcParams.maxBitRate = if config.max_bitrate > 0 {
                config.max_bitrate
            } else {
                config.bitrate * 3 / 2
            };
        }
        // If bitrate == 0, the preset default (typically CQP) is used.

        let mut init_params: NV_ENC_INITIALIZE_PARAMS = unsafe { std::mem::zeroed() };
        init_params.version = nvenc_struct_version(1);
        init_params.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
        init_params.presetGUID = NV_ENC_PRESET_P7_GUID;
        init_params.encodeWidth = config.width;
        init_params.encodeHeight = config.height;
        init_params.darWidth = config.width;
        init_params.darHeight = config.height;
        init_params.frameRateNum = config.fps_num;
        init_params.frameRateDen = config.fps_den;
        init_params.enablePTD = 1; // Enable picture-type decision.
        init_params.encodeConfig = &mut enc_config;
        init_params.tuningInfo = NV_ENC_TUNING_INFO::HIGH_QUALITY;
        init_params.maxEncodeWidth = config.width;
        init_params.maxEncodeHeight = config.height;

        let init_fn = fns
            .nvEncInitializeEncoder
            .ok_or_else(|| EngineError::Encode("nvEncInitializeEncoder not found".into()))?;

        // SAFETY: init_params points to valid config with encodeConfig set.
        unsafe {
            check_nvenc(init_fn(encoder, &mut init_params), "nvEncInitializeEncoder")?;
        }

        info!("NVENC encoder initialized — HEVC P7 High Quality");

        // ── Create bitstream buffer ──
        let create_bs_fn = fns
            .nvEncCreateBitstreamBuffer
            .ok_or_else(|| EngineError::Encode("nvEncCreateBitstreamBuffer not found".into()))?;

        let mut bs_params: NV_ENC_CREATE_BITSTREAM_BUFFER = unsafe { std::mem::zeroed() };
        bs_params.version = nvenc_struct_version(1);

        // SAFETY: bs_params is initialized. NVENC allocates the buffer.
        unsafe {
            check_nvenc(
                create_bs_fn(encoder, &mut bs_params),
                "nvEncCreateBitstreamBuffer",
            )?;
        }

        let bitstream_buf = bs_params.bitstreamBuffer;
        debug!("NVENC bitstream buffer created");

        Ok(Self {
            encoder,
            fns,
            bitstream_buf,
            sink,
            reg_cache: RegistrationCache::new(),
            config,
            frame_idx: 0,
        })
    }

    /// Register a CUDA device pointer as an NVENC input resource.
    fn register_resource(&mut self, dev_ptr: u64, pitch: u32) -> Result<*mut c_void> {
        let reg_fn = self
            .fns
            .nvEncRegisterResource
            .ok_or_else(|| EngineError::Encode("nvEncRegisterResource not found".into()))?;

        let mut reg: NV_ENC_REGISTER_RESOURCE = unsafe { std::mem::zeroed() };
        reg.version = nvenc_struct_version(1);
        reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE::CUDADEVICEPTR;
        reg.width = self.config.width;
        reg.height = self.config.height;
        reg.pitch = pitch;
        reg.resourceToRegister = dev_ptr as *mut c_void;
        reg.bufferFormat = NV_ENC_BUFFER_FORMAT::NV12;

        // SAFETY: reg is fully initialized. NVENC validates the device pointer.
        unsafe {
            check_nvenc(reg_fn(self.encoder, &mut reg), "nvEncRegisterResource")?;
        }

        let handle = reg.registeredResource;
        self.reg_cache.insert(dev_ptr, handle);
        debug!(dev_ptr, "NVENC resource registered");
        Ok(handle)
    }

    /// Encode a single frame from a registered CUDA resource.
    fn encode_frame(&mut self, frame: &FrameEnvelope) -> Result<()> {
        let dev_ptr = frame.texture.device_ptr();
        let pitch = frame.texture.pitch as u32;

        // Get or create registration.
        let reg_handle = match self.reg_cache.get(dev_ptr) {
            Some(h) => h,
            None => self.register_resource(dev_ptr, pitch)?,
        };

        // Map input resource.
        let map_fn = self
            .fns
            .nvEncMapInputResource
            .ok_or_else(|| EngineError::Encode("nvEncMapInputResource not found".into()))?;

        let mut map_params: NV_ENC_MAP_INPUT_RESOURCE = unsafe { std::mem::zeroed() };
        map_params.version = nvenc_struct_version(1);
        map_params.registeredResource = reg_handle;

        // SAFETY: reg_handle is valid (from nvEncRegisterResource).
        unsafe {
            check_nvenc(
                map_fn(self.encoder, &mut map_params),
                "nvEncMapInputResource",
            )?;
        }

        let mapped_resource = map_params.mappedResource;

        // Encode.
        let encode_fn = self
            .fns
            .nvEncEncodePicture
            .ok_or_else(|| EngineError::Encode("nvEncEncodePicture not found".into()))?;

        let mut pic_params: NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        pic_params.version = nvenc_struct_version(1);
        pic_params.inputWidth = self.config.width;
        pic_params.inputHeight = self.config.height;
        pic_params.inputPitch = pitch;
        pic_params.inputBuffer = mapped_resource;
        pic_params.outputBitstream = self.bitstream_buf;
        pic_params.bufferFmt = NV_ENC_BUFFER_FORMAT::NV12;
        pic_params.pictureStruct = NV_ENC_PIC_STRUCT::FRAME;
        pic_params.frameIdx = self.frame_idx;
        pic_params.inputTimeStamp = frame.pts as u64;

        // Force IDR on keyframes from the pipeline.
        if frame.is_keyframe {
            pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
        }

        // SAFETY: All pointers are valid NVENC-owned handles.
        unsafe {
            check_nvenc(
                encode_fn(self.encoder, &mut pic_params),
                "nvEncEncodePicture",
            )?;
        }

        self.frame_idx += 1;

        // Lock bitstream output.
        let lock_fn = self
            .fns
            .nvEncLockBitstream
            .ok_or_else(|| EngineError::Encode("nvEncLockBitstream not found".into()))?;
        let unlock_fn = self
            .fns
            .nvEncUnlockBitstream
            .ok_or_else(|| EngineError::Encode("nvEncUnlockBitstream not found".into()))?;

        let mut lock_params: NV_ENC_LOCK_BITSTREAM = unsafe { std::mem::zeroed() };
        lock_params.version = nvenc_struct_version(1);
        lock_params.outputBitstream = self.bitstream_buf;

        // SAFETY: bitstream_buf is valid (from nvEncCreateBitstreamBuffer).
        unsafe {
            check_nvenc(
                lock_fn(self.encoder, &mut lock_params),
                "nvEncLockBitstream",
            )?;
        }

        // Copy encoded data to sink.
        let is_idr = matches!(lock_params.pictureType, NV_ENC_PIC_TYPE::IDR);
        let data = unsafe {
            // SAFETY: bitstreamBufferPtr is valid for bitstreamSizeInBytes.
            std::slice::from_raw_parts(
                lock_params.bitstreamBufferPtr as *const u8,
                lock_params.bitstreamSizeInBytes as usize,
            )
        };

        let sink_result = self.sink.write_packet(data, frame.pts, is_idr);

        // Unlock regardless of sink result.
        // SAFETY: bitstream_buf was locked above.
        unsafe {
            check_nvenc(
                unlock_fn(self.encoder, self.bitstream_buf),
                "nvEncUnlockBitstream",
            )?;
        }

        // Unmap input resource.
        let unmap_fn = self
            .fns
            .nvEncUnmapInputResource
            .ok_or_else(|| EngineError::Encode("nvEncUnmapInputResource not found".into()))?;

        // SAFETY: mapped_resource is valid (from nvEncMapInputResource).
        unsafe {
            check_nvenc(
                unmap_fn(self.encoder, mapped_resource),
                "nvEncUnmapInputResource",
            )?;
        }

        sink_result
    }

    /// Send EOS to flush the encoder.
    fn send_eos(&mut self) -> Result<()> {
        let encode_fn = self
            .fns
            .nvEncEncodePicture
            .ok_or_else(|| EngineError::Encode("nvEncEncodePicture not found".into()))?;

        let mut eos_params: NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        eos_params.version = nvenc_struct_version(1);
        eos_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;

        // SAFETY: EOS params with no input buffer — signals end of encode.
        unsafe {
            check_nvenc(
                encode_fn(self.encoder, &mut eos_params),
                "nvEncEncodePicture (EOS)",
            )?;
        }

        Ok(())
    }
}

impl FrameEncoder for NvEncoder {
    fn encode(&mut self, frame: FrameEnvelope) -> Result<()> {
        if frame.texture.format != PixelFormat::Nv12 {
            return Err(EngineError::FormatMismatch {
                expected: PixelFormat::Nv12,
                actual: frame.texture.format,
            });
        }
        self.encode_frame(&frame)
    }

    fn flush(&mut self) -> Result<()> {
        self.send_eos()?;
        self.sink.flush()?;
        info!(frames = self.frame_idx, "NVENC encoder flushed");
        Ok(())
    }
}

impl Drop for NvEncoder {
    fn drop(&mut self) {
        // Unregister all cached resources.
        if let Some(unreg_fn) = self.fns.nvEncUnregisterResource {
            for handle in self.reg_cache.handles().collect::<Vec<_>>() {
                // SAFETY: handle was registered via nvEncRegisterResource.
                unsafe {
                    unreg_fn(self.encoder, handle);
                }
            }
        }

        // Destroy bitstream buffer.
        if !self.bitstream_buf.is_null() {
            if let Some(destroy_fn) = self.fns.nvEncDestroyBitstreamBuffer {
                // SAFETY: bitstream_buf was created via nvEncCreateBitstreamBuffer.
                unsafe {
                    destroy_fn(self.encoder, self.bitstream_buf);
                }
            }
        }

        // Destroy encoder session.
        if !self.encoder.is_null() {
            if let Some(destroy_fn) = self.fns.nvEncDestroyEncoder {
                // SAFETY: encoder was opened via nvEncOpenEncodeSessionEx.
                unsafe {
                    destroy_fn(self.encoder);
                }
            }
        }

        debug!("NVENC encoder destroyed");
    }
}
