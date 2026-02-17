//! GPU-resident frame types and pixel format contracts.
//!
//! # Ownership model
//!
//! [`GpuTexture`] wraps device memory via `Arc<CudaSlice<u8>>`.  This provides:
//!
//! - **RAII**: device memory is freed when the last `Arc` reference drops.
//!   `CudaSlice` stores an internal `Arc<CudaDevice>` so the CUDA context
//!   outlives all allocations automatically.
//! - **Send + Sync**: `CudaSlice<u8>` is `Send + Sync` (CUDA driver API is
//!   thread-safe for distinct streams). The `Arc` wrapper preserves these bounds.
//! - **Cheap clone**: cloning a `GpuTexture` increments a reference count.
//!   No device memory is copied.
//!
//! # Invariants
//!
//! 1. `data` **always** points to device-resident memory allocated through the
//!    engine's shared `CudaDevice`.  It is never backed by host pinned memory,
//!    managed memory, or system RAM.
//! 2. `data.len()` ≥ `format.byte_size(width, height, pitch)`.
//! 3. The `CudaDevice` that allocated `data` must not be destroyed before
//!    the `GpuTexture` is dropped.  This is guaranteed structurally because
//!    `CudaSlice` holds an `Arc<CudaDevice>`.

use cudarc::driver::{CudaSlice, DevicePtr};
use std::sync::Arc;

// ─── Pixel format ────────────────────────────────────────────────────────────

/// On-device pixel format.
///
/// Every variant documents its memory layout and byte-size formula
/// so that buffer allocation is deterministic and auditable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// **NVDEC native output.**
    /// Y plane: `height × pitch` bytes (luminance, one byte per pixel).
    /// UV plane: `(height / 2) × pitch` bytes (interleaved Cb Cr, sub-sampled 2×2).
    /// Total: `pitch × height × 3 / 2`.
    Nv12,

    /// **Inference I/O format (float32).**
    /// NCHW planar: three contiguous planes of `height × width` float32 values.
    /// Total: `3 × width × height × 4` bytes.
    /// Value range: `[0.0, 1.0]` after normalization.
    RgbPlanarF32,

    /// **Inference I/O format (float16).**
    /// Same layout as `RgbPlanarF32` but with `f16` elements.
    /// Total: `3 × width × height × 2` bytes.
    RgbPlanarF16,

    /// **Interleaved uint8 RGB.**
    /// Row-major, 3 bytes per pixel: `[R G B R G B ...]`.
    /// Total: `height × pitch` bytes (pitch ≥ `width × 3`).
    /// Used as NVENC input when the encoder accepts RGB.
    RgbInterleavedU8,
}

impl PixelFormat {
    /// Minimum device allocation size in bytes for the given dimensions.
    ///
    /// For `Nv12` and `RgbInterleavedU8`, `pitch` is the row stride in bytes
    /// (may exceed `width × bpp` for alignment).  For planar float formats
    /// the layout is dense — pitch equals `width × element_size`.
    #[inline]
    pub const fn byte_size(self, width: u32, height: u32, pitch: usize) -> usize {
        match self {
            // Y plane + UV plane (half height, same pitch)
            Self::Nv12 => pitch * (height as usize) + pitch * ((height as usize) / 2),

            // 3 dense float32 planes
            Self::RgbPlanarF32 => 3 * (width as usize) * (height as usize) * 4,

            // 3 dense float16 planes
            Self::RgbPlanarF16 => 3 * (width as usize) * (height as usize) * 2,

            // Interleaved rows with pitch
            Self::RgbInterleavedU8 => pitch * (height as usize),
        }
    }

    /// Bytes per channel-element (1 for u8, 2 for f16, 4 for f32).
    #[inline]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::Nv12 | Self::RgbInterleavedU8 => 1,
            Self::RgbPlanarF16 => 2,
            Self::RgbPlanarF32 => 4,
        }
    }

    /// Number of channels in the format.
    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            // NV12 is not channel-separable in the RGB sense, but has Y + UV
            Self::Nv12 => 1, // treat as single-channel for the Y plane
            Self::RgbPlanarF32 | Self::RgbPlanarF16 | Self::RgbInterleavedU8 => 3,
        }
    }
}

// ─── GpuTexture ──────────────────────────────────────────────────────────────

/// A single video frame residing entirely in GPU device memory.
///
/// # Thread safety
///
/// `GpuTexture` is `Send + Sync`.  Multiple pipeline stages may hold
/// `Arc`-cloned references concurrently for read-only access.  Mutation
/// requires exclusive ownership (unwrapping the `Arc` or allocating a
/// new buffer).
///
/// # Drop behavior
///
/// When the last clone is dropped, `CudaSlice::drop` calls
/// `cuMemFree` on the device pointer.  No explicit cleanup is required.
#[derive(Clone, Debug)]
pub struct GpuTexture {
    /// Device-resident memory.  Never host-backed.
    pub data: Arc<CudaSlice<u8>>,

    /// Frame width in pixels.
    pub width: u32,

    /// Frame height in pixels.
    pub height: u32,

    /// Row pitch (stride) in bytes.  For dense planar formats this equals
    /// `width × element_bytes`; for NV12/interleaved it may be larger
    /// for CUDA alignment (typically 256-byte aligned by NVDEC).
    pub pitch: usize,

    /// Pixel storage format.
    pub format: PixelFormat,
}

// Compile-time proof that GpuTexture is Send + Sync.
// CudaSlice<u8> is Send + Sync in cudarc, and Arc preserves both.
#[allow(dead_code)]
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<GpuTexture>();
    }
};

impl GpuTexture {
    /// Raw CUDA device pointer as a `u64` (suitable for FFI / kernel args).
    ///
    /// # Safety contract
    ///
    /// The returned pointer is valid only while `self` (or any clone) is alive.
    /// Callers must not free the pointer or pass it to a different CUDA context.
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        *self.data.device_ptr() as u64
    }

    /// Total byte size of the device allocation.
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.format.byte_size(self.width, self.height, self.pitch)
    }

    /// Byte offset to the UV plane for NV12 textures.
    /// Panics if `self.format != PixelFormat::Nv12`.
    #[inline]
    pub fn nv12_uv_offset(&self) -> usize {
        assert_eq!(
            self.format,
            PixelFormat::Nv12,
            "nv12_uv_offset called on non-NV12 texture"
        );
        self.pitch * (self.height as usize)
    }

    /// Stride in elements between NCHW channels for planar float formats.
    /// Returns `width × height` (element count per plane).
    #[inline]
    pub fn channel_stride_elements(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

// ─── Frame envelope ──────────────────────────────────────────────────────────

/// A GPU-resident frame annotated with pipeline metadata.
///
/// Travels through bounded channels between pipeline stages.
/// Ordering is maintained by the FIFO channel semantics.
#[derive(Clone, Debug)]
pub struct FrameEnvelope {
    /// The GPU-resident frame data.
    pub texture: GpuTexture,

    /// Zero-based frame index within the current video.
    pub frame_index: u64,

    /// Presentation timestamp in the source container's time base.
    /// Preserved through the pipeline for correct output timing.
    pub pts: i64,

    /// Whether the source frame was a keyframe (informational).
    pub is_keyframe: bool,
}

// Sentinel envelope that signals end-of-stream (EOS) through pipeline channels.
// When a stage receives `None` from its input channel, it must:
// 1. Flush any buffered state.
// 2. Drop its output sender to propagate EOS downstream.
// 3. Return cleanly.
//
// This type is not used directly — we encode EOS as channel closure (sender drop).
// The comment exists to document the protocol.
//
// ```text
// Decoder closes tx → Preprocess sees None → closes its tx → … → Encoder returns
// ```
