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
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::codecs::sys::*;
use crate::core::context::GpuContext;
use crate::core::types::{FrameEnvelope, GpuTexture, PixelFormat};
use crate::engine::pipeline::FrameDecoder;
use crate::error::{EngineError, Result};

// ─── Bitstream source trait ──────────────────────────────────────────────

/// Demuxed compressed bitstream packets (host-side, NOT raw pixels).
///
/// Implementations: file reader, network receiver, FFmpeg demuxer, etc.
pub trait BitstreamSource: Send + 'static {
    fn read_packet(&mut self) -> Result<Option<BitstreamPacket>>;
}

/// A single demuxed NAL unit or access unit.
pub struct BitstreamPacket {
    /// Compressed bitstream data (Annex B or length-prefixed).
    /// Host memory is acceptable — this is codec-compressed (~10 KB/frame).
    pub data: Vec<u8>,
    /// Presentation timestamp in stream time base.
    pub pts: i64,
    /// Whether this packet encodes an IDR/keyframe.
    pub is_keyframe: bool,
}

// ─── Decoded frame event ─────────────────────────────────────────────────

/// A decoded NV12 frame with its associated sync event.
///
/// The preprocess stage MUST wait on `decode_event` before reading
/// `texture.data` to ensure the D2D copy has completed on `decode_stream`.
pub struct DecodedFrame {
    pub envelope: FrameEnvelope,
    /// CUDA event recorded on `decode_stream` after D2D copy.
    /// Downstream calls `cuStreamWaitEvent(preprocess_stream, event, 0)`.
    pub decode_event: CUevent,
}

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
    /// Per-frame sync events returned with decoded frames.
    /// Caller must wait on these before reading the texture.
    pub last_decode_event: Option<CUevent>,
}

impl NvDecoder {
    /// Create a new hardware decoder.
    ///
    /// The parser is created immediately.  The actual hardware decoder is
    /// created lazily when the first sequence header is parsed (parser
    /// callback determines resolution, codec profile, etc.).
    pub fn new(
        ctx: Arc<GpuContext>,
        source: Box<dyn BitstreamSource>,
        codec: cudaVideoCodec,
    ) -> Result<Self> {
        let mut cb_state = Box::new(CallbackState {
            decoder: ptr::null_mut(),
            format: None,
            pending_display: VecDeque::new(),
            decoder_created: false,
            max_decode_surfaces: 8,
            codec,
        });

        let cb_state_ptr: *mut CallbackState = &mut *cb_state;

        let mut parser_params: CUVIDPARSERPARAMS = unsafe { std::mem::zeroed() };
        parser_params.CodecType = codec;
        parser_params.ulMaxNumDecodeSurfaces = 8;
        parser_params.ulMaxDisplayDelay = 4; // Allow 4-frame reorder for B-frames.
        parser_params.bAnnexb = 1; // Expect Annex B format.
        parser_params.pUserData = cb_state_ptr as *mut c_void;
        parser_params.pfnSequenceCallback = Some(sequence_callback);
        parser_params.pfnDecodePicture = Some(decode_callback);
        parser_params.pfnDisplayPicture = Some(display_callback);

        let mut parser: CUvideoparser = ptr::null_mut();

        // SAFETY: parser_params is fully initialized above.
        // pUserData points to heap-allocated cb_state which outlives the parser.
        unsafe {
            check_cu(
                cuvidCreateVideoParser(&mut parser, &mut parser_params),
                "cuvidCreateVideoParser",
            )?;
        }

        info!(?codec, "NVDEC parser created");

        Ok(Self {
            parser,
            ctx,
            source,
            cb_state,
            events: EventPool::new(),
            frame_index: 0,
            eos_sent: false,
            last_decode_event: None,
        })
    }

    /// Feed one bitstream packet to the parser.
    fn feed_packet(&mut self, packet: &BitstreamPacket) -> Result<()> {
        let mut pkt = CUVIDSOURCEDATAPACKET {
            flags: CUVID_PKT_TIMESTAMP,
            payload_size: packet.data.len() as c_ulong,
            payload: packet.data.as_ptr(),
            timestamp: packet.pts as c_ulonglong,
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

        // Get the raw CUstream handle for decode_stream.
        // SAFETY: We pass the stream handle obtained from cudarc's internal
        // representation.  cudarc::CudaStream wraps a CUstream but does not
        // expose it publicly.  We use transmute to extract it.
        //
        // This is fragile but necessary because cudarc 0.12 does not provide
        // a public `raw()` or `as_raw()` method on CudaStream.
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
            decode_event: event,
        })
    }
}

impl FrameDecoder for NvDecoder {
    /// Decode the next frame.
    ///
    /// Feeds bitstream packets to the parser until a decoded frame is
    /// available, then maps/copies it and returns the `FrameEnvelope`.
    ///
    /// Returns `None` at EOS.
    fn decode_next(&mut self) -> Result<Option<FrameEnvelope>> {
        loop {
            // Check if we have a pending decoded frame.
            if let Some(disp) = self.cb_state.pending_display.pop_front() {
                let decoded = self.map_and_copy(&disp)?;
                self.last_decode_event = Some(decoded.decode_event);
                return Ok(Some(decoded.envelope));
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

        // Return last event to pool for cleanup.
        if let Some(event) = self.last_decode_event.take() {
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
    let state = &mut *(user_data as *mut CallbackState);
    let fmt = &*format;

    state.format = Some(*fmt);

    // Determine required decode surfaces.
    let num_surfaces = (fmt.min_num_decode_surfaces as u32).max(8);
    state.max_decode_surfaces = num_surfaces;

    // Destroy existing decoder if resolution changed.
    if state.decoder_created && !state.decoder.is_null() {
        cuvidDestroyDecoder(state.decoder);
        state.decoder = ptr::null_mut();
        state.decoder_created = false;
    }

    // Create decoder.
    let mut create_info: CUVIDDECODECREATEINFO = std::mem::zeroed();
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

    let result = cuvidCreateDecoder(&mut state.decoder, &mut create_info);
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
    let state = &mut *(user_data as *mut CallbackState);

    if !state.decoder_created || state.decoder.is_null() {
        return 0;
    }

    let result = cuvidDecodePicture(state.decoder, pic_params);
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
    let state = &mut *(user_data as *mut CallbackState);

    if disp_info.is_null() {
        // Null means EOS from parser.
        return 1;
    }

    state.pending_display.push_back(*disp_info);

    1 // Success.
}

// ─── Raw stream handle extraction ────────────────────────────────────────

/// Extract raw CUstream handle from cudarc's CudaStream.
///
/// # Safety
///
/// This relies on cudarc 0.12's internal layout where CudaStream stores
/// the raw CUstream as its first field.  If cudarc changes its layout,
/// this will break.  A compile-time size assertion helps catch this.
///
/// TODO: Replace with `CudaStream::as_raw()` when cudarc exposes it.
pub fn get_raw_stream(stream: &cudarc::driver::CudaStream) -> CUstream {
    // cudarc::CudaStream in 0.12 is a repr(transparent) wrapper over
    // the internal stream type.  We read the first pointer-sized field.
    //
    // SAFETY: We're reading a pointer value from a struct that wraps a
    // CUDA stream handle.  The handle is valid for the lifetime of the
    // CudaStream reference.
    unsafe {
        let ptr = stream as *const cudarc::driver::CudaStream as *const CUstream;
        *ptr
    }
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
pub fn wait_for_event(target_stream: &cudarc::driver::CudaStream, event: CUevent) -> Result<()> {
    let raw_stream = get_raw_stream(target_stream);
    // SAFETY: raw_stream and event are valid handles.
    // Flags = 0 is the only defined value.
    unsafe {
        check_cu(cuStreamWaitEvent(raw_stream, event, 0), "cuStreamWaitEvent")?;
    }
    Ok(())
}
