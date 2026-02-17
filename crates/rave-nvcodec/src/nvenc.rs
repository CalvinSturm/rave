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
use std::{fs, mem};

use tracing::{debug, info, warn};

use crate::sys::*;
use rave_core::codec_traits::{BitstreamSink, FrameEncoder};
use rave_core::error::{EngineError, Result};
use rave_core::types::{FrameEnvelope, PixelFormat};

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

struct CudaContextGuard {
    prev_ctx: CUcontext,
    restore: bool,
}

impl CudaContextGuard {
    fn make_current(target_ctx: CUcontext) -> Result<Self> {
        let mut prev_ctx: CUcontext = ptr::null_mut();
        unsafe {
            check_cu(cuCtxGetCurrent(&mut prev_ctx), "cuCtxGetCurrent")?;
        }
        if prev_ctx != target_ctx {
            unsafe {
                check_cu(cuCtxSetCurrent(target_ctx), "cuCtxSetCurrent")?;
            }
        }
        Ok(Self {
            prev_ctx,
            restore: prev_ctx != target_ctx,
        })
    }
}

impl Drop for CudaContextGuard {
    fn drop(&mut self) {
        if self.restore {
            let rc = unsafe { cuCtxSetCurrent(self.prev_ctx) };
            if rc != CUDA_SUCCESS {
                warn!(rc, "cuCtxSetCurrent restore failed");
            }
        }
    }
}

fn ptr_hex(ptr: *mut c_void) -> String {
    format!("{ptr:p}")
}

fn nvenc_version_parts(v: u32) -> (u32, u32) {
    (v & 0x00ff_ffff, (v >> 24) & 0xff)
}

fn resolved_libnvidia_encode_path() -> Option<String> {
    let maps = fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines()
        .find_map(|line| line.split_whitespace().last())
        .filter(|p| p.contains("libnvidia-encode.so"))
        .map(|s| s.to_string())
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
    /// CUDA context handle used for all NVENC API calls.
    cuda_context: CUcontext,
    /// Frame counter for encode ordering.
    frame_idx: u32,
    /// Monotonic DTS counter (starts at -b_frames to give B-frame lookahead room).
    dts_counter: i64,
    /// Duration of one frame in microseconds.
    frame_duration_us: i64,
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
        let target_cuda_ctx = cuda_context as CUcontext;
        let mut api_version = NVENCAPI_VERSION;
        let mut max_supported_version = 0u32;
        unsafe {
            check_nvenc(
                NvEncodeAPIGetMaxSupportedVersion(&mut max_supported_version),
                "NvEncodeAPIGetMaxSupportedVersion",
            )?;
        }
        let (compiled_major, compiled_minor) = nvenc_version_parts(NVENCAPI_VERSION);
        let (runtime_major, runtime_minor) = nvenc_version_parts(max_supported_version);
        info!(
            libnvidia_encode = ?resolved_libnvidia_encode_path(),
            compiled_api_major = compiled_major,
            compiled_api_minor = compiled_minor,
            runtime_max_api_major = runtime_major,
            runtime_max_api_minor = runtime_minor,
            "NVENC runtime environment"
        );
        if api_version > max_supported_version {
            info!(
                requested_api_version = api_version,
                max_supported_version = max_supported_version,
                "NVENC API version exceeds driver support; downgrading open-session apiVersion"
            );
            api_version = max_supported_version;
        } else {
            info!(
                requested_api_version = api_version,
                max_supported_version = max_supported_version,
                "NVENC max supported version queried"
            );
        }

        // ── Get function table ──
        let mut fns: NV_ENCODE_API_FUNCTION_LIST = unsafe { std::mem::zeroed() };
        fns.version = NV_ENCODE_API_FUNCTION_LIST_VER;

        // SAFETY: fns is zeroed with version set.
        // NvEncodeAPICreateInstance fills the function pointers.
        unsafe {
            check_nvenc(
                NvEncodeAPICreateInstance(&mut fns),
                "NvEncodeAPICreateInstance",
            )?;
        }
        info!(
            function_list_version = format_args!("{:#010x}", fns.version),
            open_session_ex_is_null = fns.nvEncOpenEncodeSessionEx.is_none(),
            get_encode_guid_count_is_null = fns.nvEncGetEncodeGUIDCount.is_null(),
            get_encode_guids_is_null = fns.nvEncGetEncodeGUIDs.is_null(),
            "NVENC function list initialized"
        );
        if fns.nvEncOpenEncodeSessionEx.is_none() {
            return Err(EngineError::Encode(
                "nvEncOpenEncodeSessionEx missing in function list".into(),
            ));
        }
        if fns.nvEncGetEncodeGUIDCount.is_null() {
            return Err(EngineError::Encode(
                "nvEncGetEncodeGUIDCount missing in function list".into(),
            ));
        }
        if fns.nvEncGetEncodeGUIDs.is_null() {
            return Err(EngineError::Encode(
                "nvEncGetEncodeGUIDs missing in function list".into(),
            ));
        }

        // ── Open session ──
        let mut open_params: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS = unsafe { std::mem::zeroed() };
        open_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        open_params.deviceType = NV_ENC_DEVICE_TYPE::CUDA;
        open_params.device = target_cuda_ctx;
        open_params.apiVersion = api_version;

        let mut encoder: *mut c_void = ptr::null_mut();
        let open_fn = fns
            .nvEncOpenEncodeSessionEx
            .ok_or_else(|| EngineError::Encode("nvEncOpenEncodeSessionEx not found".into()))?;

        let mut current_ctx: CUcontext = ptr::null_mut();
        unsafe {
            check_cu(
                cuCtxGetCurrent(&mut current_ctx),
                "cuCtxGetCurrent (before nvEncOpenEncodeSessionEx)",
            )?;
        }
        info!(
            thread_id = ?std::thread::current().id(),
            open_params_version = format_args!("{:#010x}", open_params.version),
            open_params_api_version = open_params.apiVersion,
            device_type = ?open_params.deviceType,
            device_type_raw = open_params.deviceType as u32,
            open_device_ptr = %ptr_hex(open_params.device),
            current_cuda_ctx = %ptr_hex(current_ctx),
            target_cuda_ctx = %ptr_hex(target_cuda_ctx),
            "NVENC open-session diagnostics"
        );
        let _ctx_guard = CudaContextGuard::make_current(target_cuda_ctx)?;

        // SAFETY: open_params is fully initialized.
        let open_status = unsafe { open_fn(&mut open_params, &mut encoder) };
        if open_status != NV_ENC_SUCCESS {
            info!(
                status = open_status,
                status_name = nvenc_status_name(open_status),
                "nvEncOpenEncodeSessionEx failed"
            );
        }
        check_nvenc(open_status, "nvEncOpenEncodeSessionEx")?;

        info!(
            width = config.width,
            height = config.height,
            fps = format!("{}/{}", config.fps_num, config.fps_den),
            bitrate = config.bitrate,
            "NVENC session opened"
        );

        // ── Get preset config ──
        type GetPresetFn =
            unsafe extern "C" fn(*mut c_void, GUID, GUID, *mut NV_ENC_PRESET_CONFIG) -> NVENCSTATUS;
        let get_preset_fn = if fns.nvEncGetEncodePresetConfig.is_null() {
            None
        } else {
            // SAFETY: Function pointer layout matches NVENC API declaration.
            Some(unsafe {
                mem::transmute::<*const c_void, GetPresetFn>(fns.nvEncGetEncodePresetConfig)
            })
        };

        let mut preset_config: NV_ENC_PRESET_CONFIG = unsafe { std::mem::zeroed() };
        let mut got_preset = false;
        let version_candidates = [8, 7, 6, 5, 4, 3, 2, 1];

        if let Some(get_preset_ex_fn) = fns.nvEncGetEncodePresetConfigEx {
            for ver in version_candidates {
                preset_config = unsafe { std::mem::zeroed() };
                preset_config.version = nvenc_struct_version(ver);
                preset_config.presetCfg.version = nvenc_struct_version(ver);
                let status = unsafe {
                    get_preset_ex_fn(
                        encoder,
                        NV_ENC_CODEC_HEVC_GUID,
                        NV_ENC_PRESET_P7_GUID,
                        NV_ENC_TUNING_INFO::HIGH_QUALITY,
                        &mut preset_config,
                    )
                };
                if status == NV_ENC_SUCCESS {
                    got_preset = true;
                    info!(
                        struct_version = format_args!("{:#010x}", preset_config.version),
                        "Loaded NVENC preset via nvEncGetEncodePresetConfigEx"
                    );
                    break;
                }
                if status != NV_ENC_ERR_INVALID_VERSION {
                    check_nvenc(status, "nvEncGetEncodePresetConfigEx")?;
                }
            }
        }

        if !got_preset && let Some(get_preset_legacy_fn) = get_preset_fn {
            for ver in version_candidates {
                preset_config = unsafe { std::mem::zeroed() };
                preset_config.version = nvenc_struct_version(ver);
                preset_config.presetCfg.version = nvenc_struct_version(ver);
                let status = unsafe {
                    get_preset_legacy_fn(
                        encoder,
                        NV_ENC_CODEC_HEVC_GUID,
                        NV_ENC_PRESET_P7_GUID,
                        &mut preset_config,
                    )
                };
                if status == NV_ENC_SUCCESS {
                    got_preset = true;
                    info!(
                        struct_version = format_args!("{:#010x}", preset_config.version),
                        "Loaded NVENC preset via legacy nvEncGetEncodePresetConfig"
                    );
                    break;
                }
                if status != NV_ENC_ERR_INVALID_VERSION {
                    check_nvenc(status, "nvEncGetEncodePresetConfig")?;
                }
            }
        }

        // ── Configure encoder ──
        let mut enc_config = if got_preset {
            preset_config.presetCfg
        } else {
            warn!(
                "Could not query NVENC preset config due to version mismatch; using default encoder config"
            );
            unsafe { std::mem::zeroed() }
        };
        if enc_config.version == 0 {
            enc_config.version = nvenc_struct_version(8);
        }
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
        init_params.version = nvenc_struct_version(8);
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

        // Retry with older structure versions if runtime rejects the compiled one.
        let mut init_ok = false;
        for ver in version_candidates {
            init_params.version = nvenc_struct_version(ver);
            let status = unsafe { init_fn(encoder, &mut init_params) };
            if status == NV_ENC_SUCCESS {
                init_ok = true;
                break;
            }
            if status != NV_ENC_ERR_INVALID_VERSION {
                check_nvenc(status, "nvEncInitializeEncoder")?;
            }
        }
        if !init_ok {
            return Err(EngineError::Encode(
                "nvEncInitializeEncoder: all struct-version retries returned NV_ENC_ERR_INVALID_VERSION".into(),
            ));
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

        let dts_counter = -(config.b_frames as i64);
        let frame_duration_us = 1_000_000 * config.fps_den as i64 / config.fps_num as i64;

        Ok(Self {
            encoder,
            fns,
            bitstream_buf,
            sink,
            reg_cache: RegistrationCache::new(),
            config,
            cuda_context: target_cuda_ctx,
            frame_idx: 0,
            dts_counter,
            frame_duration_us,
        })
    }

    /// Register a CUDA device pointer as an NVENC input resource.
    fn register_resource(&mut self, dev_ptr: u64, pitch: u32) -> Result<*mut c_void> {
        let _ctx_guard = CudaContextGuard::make_current(self.cuda_context)?;
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
        let _ctx_guard = CudaContextGuard::make_current(self.cuda_context)?;
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
        let output_pts = lock_params.outputTimeStamp as i64;
        let dts = self.dts_counter * self.frame_duration_us;
        self.dts_counter += 1;

        let data = unsafe {
            // SAFETY: bitstreamBufferPtr is valid for bitstreamSizeInBytes.
            std::slice::from_raw_parts(
                lock_params.bitstreamBufferPtr as *const u8,
                lock_params.bitstreamSizeInBytes as usize,
            )
        };

        let sink_result = self.sink.write_packet(data, output_pts, dts, is_idr);

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
        let _ctx_guard = CudaContextGuard::make_current(self.cuda_context)?;
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
        let _ctx_guard = CudaContextGuard::make_current(self.cuda_context).ok();

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
        if !self.bitstream_buf.is_null()
            && let Some(destroy_fn) = self.fns.nvEncDestroyBitstreamBuffer
        {
            // SAFETY: bitstream_buf was created via nvEncCreateBitstreamBuffer.
            unsafe {
                destroy_fn(self.encoder, self.bitstream_buf);
            }
        }

        // Destroy encoder session.
        if !self.encoder.is_null()
            && let Some(destroy_fn) = self.fns.nvEncDestroyEncoder
        {
            // SAFETY: encoder was opened via nvEncOpenEncodeSessionEx.
            unsafe {
                destroy_fn(self.encoder);
            }
        }

        debug!("NVENC encoder destroyed");
    }
}
