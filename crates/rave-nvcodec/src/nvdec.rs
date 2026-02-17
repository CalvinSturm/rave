//! NVDEC hardware decoder — zero-copy GPU-resident NV12 output.
//!
//! # Architecture
//!
//! ```text
//! BitstreamSource ──(host NAL)──▸ cuvidParseVideoData
//!                                       │
//!                        ┌───callback────┘
//!                        ▼
//!              cuvidDecodePicture (on NVDEC HW)
//!                        │
//!              cuvidMapVideoFrame64
//!                        │
//!           ┌── NVDEC surface (device ptr + pitch) ──┐
//!           │                                        │
//!           │  cuMemcpy2DAsync (D2D on decode_stream)│
//!           │                                        │
//!           └── our CudaSlice buffer ────────────────┘
//!                        │
//!              cuvidUnmapVideoFrame64
//!                        │
//!              cuEventRecord(decode_done, decode_stream)
//!                        │
//!              GpuTexture { NV12, pitch-aligned }
//! ```
//!
//! # Why D2D copy?
//!
//! NVDEC surfaces are a finite pool (typically 8-16).  They must be
//! returned quickly via `cuvidUnmapVideoFrame64` to avoid stalling the
//! hardware.  Copying to our own buffer (~24 µs for 4K NV12 at 500 GB/s)
//! decouples decoder surface lifetime from pipeline frame lifetime.
//!
//! # Cross-stream synchronization
//!
//! After the D2D copy completes on `decode_stream`, a CUDA event is
//! recorded.  The preprocess stage must call
//! `cuStreamWaitEvent(preprocess_stream, event)` before reading the buffer.
//! This ensures ordering without CPU-blocking sync.

use std::collections::VecDeque;
use std::ffi::{c_int, c_short, c_uint, c_ulong, c_ulonglong, c_void};
use std::ptr;
use std::sync::Arc;

use cudarc::driver::DevicePtr;

use tracing::{debug, info};

use crate::sys::*;
use rave_core::codec_traits::{BitstreamPacket, BitstreamSource, DecodedFrame, FrameDecoder};
use rave_core::context::GpuContext;
use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec as CoreCodec;
use rave_core::types::{FrameEnvelope, GpuTexture, PixelFormat};

// ─── Per-frame event pool ────────────────────────────────────────────────

/// Reusable pool of CUDA events (avoids per-frame event creation overhead).
struct EventPool {
    free: VecDeque<CUevent>,
}

impl EventPool {
    fn new() -> Self {
        Self {
            free: VecDeque::new(),
        }
    }

    /// Get or create a CUDA event with timing disabled (lightweight).
    fn acquire(&mut self) -> Result<CUevent> {
        if let Some(event) = self.free.pop_front() {
            Ok(event)
        } else {
            let mut event: CUevent = ptr::null_mut();
            // SAFETY: cuEventCreate writes to `event`.  CU_EVENT_DISABLE_TIMING
            // avoids timing overhead — we only need ordering semantics.
            unsafe {
                check_cu(
                    cuEventCreate(&mut event, CU_EVENT_DISABLE_TIMING),
                    "cuEventCreate",
                )?;
            }
            Ok(event)
        }
    }

    /// Return an event to the pool for reuse.
    fn release(&mut self, event: CUevent) {
        self.free.push_back(event);
    }
}

impl Drop for EventPool {
    fn drop(&mut self) {
        for event in self.free.drain(..) {
            // SAFETY: event was created by cuEventCreate and is not in use.
            unsafe {
                cuEventDestroy_v2(event);
            }
        }
    }
}

// ─── Decoder callback state ─────────────────────────────────────────────

/// Shared state between parser callbacks and the main decoder.
///
/// Parser callbacks push decoded frame info here; `decode_next()` drains it.
struct CallbackState {
    decoder: CUvideodecoder,
    format: Option<CUVIDEOFORMAT>,
    pending_display: VecDeque<CUVIDPARSERDISPINFO>,
    decoder_created: bool,
    max_decode_surfaces: u32,
    codec: cudaVideoCodec,
}

// ─── NvDecoder ───────────────────────────────────────────────────────────

/// NVDEC hardware decoder producing GPU-resident NV12 `FrameEnvelope`s.
///
/// Implements [`FrameDecoder`].  Each call to `decode_next()` returns one
/// frame with a CUDA event for cross-stream synchronization.
pub struct NvDecoder {
    parser: CUvideoparser,
    ctx: Arc<GpuContext>,
    source: Box<dyn BitstreamSource>,
    cb_state: Box<CallbackState>,
    events: EventPool,
    frame_index: u64,
    eos_sent: bool,
    /// Sender half embedded in each `DecodedFrame` for event recycling.
    event_return_tx: std::sync::mpsc::Sender<CUevent>,
    /// Receiver half drained in `decode_next()` to recycle events.
    event_return_rx: std::sync::mpsc::Receiver<CUevent>,
}

unsafe impl Send for NvDecoder {}

impl NvDecoder {
    fn ensure_cuda_context_current(&self) -> Result<()> {
        let raw_ctx = *self.ctx.device().cu_primary_ctx() as CUcontext;
        // SAFETY: `raw_ctx` is a valid CUDA primary context for this device.
        unsafe { check_cu(cuCtxSetCurrent(raw_ctx), "cuCtxSetCurrent (nvdec)")? };
        Ok(())
    }

    fn ensure_parser_created(&mut self) -> Result<()> {
        if !self.parser.is_null() {
            return Ok(());
        }

        let cb_state_ptr: *mut CallbackState = &mut *self.cb_state;

        let mut parser_params: CUVIDPARSERPARAMS = unsafe { std::mem::zeroed() };
        parser_params.CodecType = self.cb_state.codec;
        parser_params.ulMaxNumDecodeSurfaces = 8;
        parser_params.ulMaxDisplayDelay = 4; // Allow 4-frame reorder for B-frames.
        parser_params.bAnnexb = 1; // Expect Annex B format.
        parser_params.pUserData = cb_state_ptr as *mut c_void;
        parser_params.pfnSequenceCallback = Some(sequence_callback);
        parser_params.pfnDecodePicture = Some(decode_callback);
        parser_params.pfnDisplayPicture = Some(display_callback);

        let mut parser: CUvideoparser = ptr::null_mut();
        // SAFETY: parser_params is fully initialized above.
        unsafe {
            check_cu(
                cuvidCreateVideoParser(&mut parser, &mut parser_params),
                "cuvidCreateVideoParser",
            )?;
        }
        self.parser = parser;
        info!(codec = ?self.cb_state.codec, "NVDEC parser created");
        Ok(())
    }

    /// Create a new hardware decoder.
    ///
    /// The parser is created immediately.  The actual hardware decoder is
    /// created lazily when the first sequence header is parsed (parser
    /// callback determines resolution, codec profile, etc.).
    pub fn new(
        ctx: Arc<GpuContext>,
        source: Box<dyn BitstreamSource>,
        codec: CoreCodec,
    ) -> Result<Self> {
        let codec = to_sys_codec(codec);
        let cb_state = Box::new(CallbackState {
            decoder: ptr::null_mut(),
            format: None,
            pending_display: VecDeque::new(),
            decoder_created: false,
            max_decode_surfaces: 8,
            codec,
        });

        let (event_return_tx, event_return_rx) = std::sync::mpsc::channel();

        Ok(Self {
            parser: ptr::null_mut(),
            ctx,
            source,
            cb_state,
            events: EventPool::new(),
            frame_index: 0,
            eos_sent: false,
            event_return_tx,
            event_return_rx,
        })
    }

    /// Feed one bitstream packet to the parser.
    fn feed_packet(&mut self, packet: &BitstreamPacket) -> Result<()> {
        if packet.data.is_empty() {
            debug!(pts = packet.pts, "Skipping empty bitstream packet");
            return Ok(());
        }

        let (flags, timestamp) = if packet.pts >= 0 {
            (CUVID_PKT_TIMESTAMP, packet.pts as c_ulonglong)
        } else {
            (0, 0)
        };

        // NVDEC parser may read a few bytes past payload end; keep explicit
        // zero padding while preserving the original payload size field.
        let payload_size = packet.data.len();
        let mut padded = Vec::with_capacity(payload_size + 64);
        padded.extend_from_slice(&packet.data);
        padded.resize(payload_size + 64, 0);

        let mut pkt = CUVIDSOURCEDATAPACKET {
            flags,
            payload_size: payload_size as c_ulong,
            payload: padded.as_ptr(),
            timestamp,
        };

        // SAFETY: pkt.payload points to valid host memory (packet.data).
        // Parser copies the data internally before returning.
        unsafe {
            check_cu(
                cuvidParseVideoData(self.parser, &mut pkt),
                "cuvidParseVideoData",
            )?;
        }
        Ok(())
    }

    /// Send EOS to the parser to flush remaining frames.
    fn send_eos(&mut self) -> Result<()> {
        let mut pkt = CUVIDSOURCEDATAPACKET {
            flags: CUVID_PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: ptr::null(),
            timestamp: 0,
        };

        // SAFETY: EOS packet has no payload.
        unsafe {
            check_cu(
                cuvidParseVideoData(self.parser, &mut pkt),
                "cuvidParseVideoData (EOS)",
            )?;
        }
        self.eos_sent = true;
        Ok(())
    }

    /// Map a decoded surface, D2D copy to our buffer, unmap, record event.
    fn map_and_copy(&mut self, disp: &CUVIDPARSERDISPINFO) -> Result<DecodedFrame> {
        let decoder = self.cb_state.decoder;
        if decoder.is_null() {
            return Err(EngineError::Decode("Decoder not created yet".into()));
        }

        let format = self
            .cb_state
            .format
            .as_ref()
            .ok_or_else(|| EngineError::Decode("No format received".into()))?;

        let width = format.coded_width;
        let height = format.coded_height;

        // Map the decoded surface to a device pointer.
        let mut src_ptr: CUdeviceptr = 0;
        let mut src_pitch: c_uint = 0;
        let mut proc_params: CUVIDPROCPARAMS = unsafe { std::mem::zeroed() };
        proc_params.progressive_frame = disp.progressive_frame;
        proc_params.top_field_first = disp.top_field_first;
        proc_params.second_field = 0;

        // SAFETY: decoder is valid (created in sequence_callback).
        // src_ptr and src_pitch are outputs.
        unsafe {
            check_cu(
                cuvidMapVideoFrame64(
                    decoder,
                    disp.picture_index,
                    &mut src_ptr,
                    &mut src_pitch,
                    &mut proc_params,
                ),
                "cuvidMapVideoFrame64",
            )?;
        }

        let src_pitch_usize = src_pitch as usize;

        // Allocate our destination buffer with the same pitch for NV12.
        // NV12 total: pitch * height * 3 / 2
        let dst_size = PixelFormat::Nv12.byte_size(width, height, src_pitch_usize);
        let dst_buf = self.ctx.alloc(dst_size)?;
        let dst_ptr = *dst_buf.device_ptr() as CUdeviceptr;

        // ── D2D copy: Y plane ──
        let y_copy = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::Device,
            srcHost: ptr::null(),
            srcDevice: src_ptr,
            srcArray: ptr::null(),
            srcPitch: src_pitch_usize,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype::Device,
            dstHost: ptr::null_mut(),
            dstDevice: dst_ptr,
            dstArray: ptr::null_mut(),
            dstPitch: src_pitch_usize,
            WidthInBytes: width as usize,
            Height: height as usize,
        };

        // ── D2D copy: UV plane ──
        let uv_src_offset = src_ptr + (src_pitch_usize * height as usize) as CUdeviceptr;
        let uv_dst_offset = dst_ptr + (src_pitch_usize * height as usize) as CUdeviceptr;

        let uv_copy = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype::Device,
            srcHost: ptr::null(),
            srcDevice: uv_src_offset,
            srcArray: ptr::null(),
            srcPitch: src_pitch_usize,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype::Device,
            dstHost: ptr::null_mut(),
            dstDevice: uv_dst_offset,
            dstArray: ptr::null_mut(),
            dstPitch: src_pitch_usize,
            WidthInBytes: width as usize,
            Height: (height / 2) as usize,
        };

        let decode_stream_raw = get_raw_stream(&self.ctx.decode_stream);

        // SAFETY: src_ptr is a valid mapped NVDEC surface.
        // dst_ptr is our allocated buffer.  Both are device memory.
        // The copy is async on decode_stream.
        unsafe {
            check_cu(
                cuMemcpy2DAsync_v2(&y_copy, decode_stream_raw),
                "cuMemcpy2DAsync_v2 (Y plane)",
            )?;
            check_cu(
                cuMemcpy2DAsync_v2(&uv_copy, decode_stream_raw),
                "cuMemcpy2DAsync_v2 (UV plane)",
            )?;
        }

        // Record event AFTER the D2D copy completes on decode_stream.
        let event = self.events.acquire()?;
        // SAFETY: event is valid, decode_stream_raw is valid.
        unsafe {
            check_cu(
                cuEventRecord(event, decode_stream_raw),
                "cuEventRecord (decode_done)",
            )?;
        }

        // Unmap the NVDEC surface — it is safe to do so because the D2D copy
        // is enqueued on decode_stream before this call.  The GPU will
        // complete the copy before reusing the surface for a new decode.
        // SAFETY: src_ptr was obtained from cuvidMapVideoFrame64.
        unsafe {
            check_cu(
                cuvidUnmapVideoFrame64(decoder, src_ptr),
                "cuvidUnmapVideoFrame64",
            )?;
        }

        let texture = GpuTexture {
            data: Arc::new(dst_buf),
            width,
            height,
            pitch: src_pitch_usize,
            format: PixelFormat::Nv12,
        };

        let envelope = FrameEnvelope {
            texture,
            frame_index: self.frame_index,
            pts: disp.timestamp as i64,
            is_keyframe: false, // Parser doesn't directly expose this.
        };

        self.frame_index += 1;

        Ok(DecodedFrame {
            envelope,
            decode_event: Some(event),
            event_return: Some(self.event_return_tx.clone()),
        })
    }
}

#[inline]
fn to_sys_codec(codec: CoreCodec) -> cudaVideoCodec {
    match codec {
        CoreCodec::MPEG1 => cudaVideoCodec::MPEG1,
        CoreCodec::MPEG2 => cudaVideoCodec::MPEG2,
        CoreCodec::MPEG4 => cudaVideoCodec::MPEG4,
        CoreCodec::VC1 => cudaVideoCodec::VC1,
        CoreCodec::H264 => cudaVideoCodec::H264,
        CoreCodec::JPEG => cudaVideoCodec::JPEG,
        CoreCodec::H264_SVC => cudaVideoCodec::H264_SVC,
        CoreCodec::H264_MVC => cudaVideoCodec::H264_MVC,
        CoreCodec::HEVC => cudaVideoCodec::HEVC,
        CoreCodec::VP8 => cudaVideoCodec::VP8,
        CoreCodec::VP9 => cudaVideoCodec::VP9,
        CoreCodec::AV1 => cudaVideoCodec::AV1,
        CoreCodec::NumCodecs => cudaVideoCodec::NumCodecs,
    }
}

impl FrameDecoder for NvDecoder {
    /// Decode the next frame.
    ///
    /// Feeds bitstream packets to the parser until a decoded frame is
    /// available, then maps/copies it and returns the `DecodedFrame` with
    /// its cross-stream sync event.
    ///
    /// Returns `None` at EOS.
    fn decode_next(&mut self) -> Result<Option<DecodedFrame>> {
        self.ensure_cuda_context_current()?;
        self.ensure_parser_created()?;

        // Drain recycled events from the preprocess stage back into the pool.
        while let Ok(event) = self.event_return_rx.try_recv() {
            self.events.release(event);
        }

        loop {
            // Check if we have a pending decoded frame.
            if let Some(disp) = self.cb_state.pending_display.pop_front() {
                let decoded = self.map_and_copy(&disp)?;
                return Ok(Some(decoded));
            }

            // No pending frame — feed more bitstream.
            if self.eos_sent {
                return Ok(None);
            }

            match self.source.read_packet()? {
                Some(packet) => {
                    self.feed_packet(&packet)?;
                }
                None => {
                    self.send_eos()?;
                    // After EOS, pending_display may have flushed frames.
                    // Loop around to check.
                }
            }
        }
    }
}

impl Drop for NvDecoder {
    fn drop(&mut self) {
        // Destroy parser first (stops callbacks).
        if !self.parser.is_null() {
            // SAFETY: parser was created by cuvidCreateVideoParser.
            unsafe {
                cuvidDestroyVideoParser(self.parser);
            }
        }

        // Destroy decoder.
        if self.cb_state.decoder_created && !self.cb_state.decoder.is_null() {
            // SAFETY: decoder was created by cuvidCreateDecoder.
            unsafe {
                cuvidDestroyDecoder(self.cb_state.decoder);
            }
        }

        // Drain any remaining recycled events back into the pool for cleanup.
        while let Ok(event) = self.event_return_rx.try_recv() {
            self.events.release(event);
        }

        debug!("NVDEC decoder destroyed");
    }
}

// ─── Parser callbacks ────────────────────────────────────────────────────
//
// These are called by cuvidParseVideoData on the calling thread.
// They must not block and must not call Rust allocator-heavy operations.

/// Called when a sequence header is parsed — creates/recreates the decoder.
unsafe extern "C" fn sequence_callback(
    user_data: *mut c_void,
    format: *mut CUVIDEOFORMAT,
) -> c_int {
    let state = unsafe { &mut *(user_data as *mut CallbackState) };
    let fmt = unsafe { &*format };

    state.format = Some(*fmt);

    // Determine required decode surfaces.
    let num_surfaces = (fmt.min_num_decode_surfaces as u32).max(8);
    state.max_decode_surfaces = num_surfaces;

    // Destroy existing decoder if resolution changed.
    if state.decoder_created && !state.decoder.is_null() {
        unsafe { cuvidDestroyDecoder(state.decoder) };
        state.decoder = ptr::null_mut();
        state.decoder_created = false;
    }

    // Create decoder.
    let mut create_info: CUVIDDECODECREATEINFO = unsafe { std::mem::zeroed() };
    create_info.ulWidth = fmt.coded_width as c_ulong;
    create_info.ulHeight = fmt.coded_height as c_ulong;
    create_info.ulNumDecodeSurfaces = num_surfaces as c_ulong;
    create_info.CodecType = state.codec;
    create_info.ChromaFormat = fmt.chroma_format;
    create_info.ulCreationFlags = cudaVideoCreateFlags::PreferCUVID as c_ulong;
    create_info.bitDepthMinus8 = fmt.bit_depth_luma_minus8 as c_ulong;
    create_info.ulIntraDecodeOnly = 0;
    create_info.ulMaxWidth = fmt.coded_width as c_ulong;
    create_info.ulMaxHeight = fmt.coded_height as c_ulong;
    create_info.display_area = CUVIDDECODECREATEINFO_display_area {
        left: 0,
        top: 0,
        right: fmt.coded_width as c_short,
        bottom: fmt.coded_height as c_short,
    };
    create_info.OutputFormat = cudaVideoSurfaceFormat::NV12;
    create_info.DeinterlaceMode = cudaVideoDeinterlaceMode::Adaptive;
    create_info.ulTargetWidth = fmt.coded_width as c_ulong;
    create_info.ulTargetHeight = fmt.coded_height as c_ulong;
    create_info.ulNumOutputSurfaces = 2;

    let result = unsafe { cuvidCreateDecoder(&mut state.decoder, &mut create_info) };
    if result != CUDA_SUCCESS {
        // Return 0 to signal failure to the parser.
        return 0;
    }

    state.decoder_created = true;

    // Return the number of decode surfaces to indicate success.
    num_surfaces as c_int
}

/// Called when a picture has been decoded — enqueue for GPU processing.
unsafe extern "C" fn decode_callback(
    user_data: *mut c_void,
    pic_params: *mut CUVIDPICPARAMS,
) -> c_int {
    let state = unsafe { &mut *(user_data as *mut CallbackState) };

    if !state.decoder_created || state.decoder.is_null() {
        return 0;
    }

    let result = unsafe { cuvidDecodePicture(state.decoder, pic_params) };
    if result != CUDA_SUCCESS {
        return 0;
    }

    1 // Success.
}

/// Called when a decoded picture is ready for display (reordered).
unsafe extern "C" fn display_callback(
    user_data: *mut c_void,
    disp_info: *mut CUVIDPARSERDISPINFO,
) -> c_int {
    let state = unsafe { &mut *(user_data as *mut CallbackState) };

    if disp_info.is_null() {
        // Null means EOS from parser.
        return 1;
    }

    state.pending_display.push_back(unsafe { *disp_info });

    1 // Success.
}

/// Get the raw CUstream handle from a cudarc CudaStream.
pub fn get_raw_stream(stream: &cudarc::driver::CudaStream) -> CUstream {
    stream.stream as CUstream
}

// ─── Stream wait helper (public for pipeline use) ────────────────────────

/// Make `target_stream` wait for `event` without blocking the CPU.
///
/// This is the cross-stream synchronization primitive.  Call this in the
/// preprocess stage before reading a decoded frame's texture data.
///
/// ```rust,ignore
/// // In preprocess stage:
/// wait_for_event(&ctx.preprocess_stream, decoded_frame.decode_event)?;
/// // Now safe to read decoded_frame.envelope.texture on preprocess_stream.
/// ```
// `event` is an opaque CUDA driver handle forwarded to cuStreamWaitEvent.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn wait_for_event(target_stream: &cudarc::driver::CudaStream, event: CUevent) -> Result<()> {
    let raw_stream = get_raw_stream(target_stream);
    // SAFETY: raw_stream and event are valid handles.
    // Flags = 0 is the only defined value.
    unsafe {
        check_cu(cuStreamWaitEvent(raw_stream, event, 0), "cuStreamWaitEvent")?;
    }
    Ok(())
}
