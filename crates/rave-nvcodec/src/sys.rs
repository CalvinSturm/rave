//! Raw FFI bindings to NVIDIA Video Codec SDK (nvcuvid + nvEncodeAPI).
//!
//! Covers the minimal subset required by [`NvDecoder`](super::nvdec) and
//! [`NvEncoder`](super::nvenc).  Matches Video Codec SDK v12.x headers.
//!
//! # Linking
//!
//! `build.rs` emits `-l nvcuvid` and `-l nvencodeapi` (Windows: `nvEncodeAPI64`).
//! Libraries are located via `CUDA_PATH` env var.
//!
//! # Safety
//!
//! All functions in this module are `unsafe extern "C"`.  The safe wrappers
//! in `nvdec.rs` and `nvenc.rs` enforce invariants documented below.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;
use std::os::raw::{c_int, c_short, c_uint, c_ulong, c_ulonglong};

// ═══════════════════════════════════════════════════════════════════════════
//  COMMON TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// CUDA result code.
pub type CUresult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;

/// CUDA device pointer (64-bit).
pub type CUdeviceptr = c_ulonglong;

/// CUDA stream handle.
pub type CUstream = *mut c_void;

/// CUDA context handle.
pub type CUcontext = *mut c_void;

// ═══════════════════════════════════════════════════════════════════════════
//  NVDEC — cuviddec.h / nvcuvid.h
// ═══════════════════════════════════════════════════════════════════════════

/// Opaque decoder handle.
pub type CUvideodecoder = *mut c_void;

/// Opaque parser handle.
pub type CUvideoparser = *mut c_void;

// ─── Enums ───────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoCodec {
    MPEG1 = 0,
    MPEG2 = 1,
    MPEG4 = 2,
    VC1 = 3,
    H264 = 4,
    JPEG = 5,
    H264_SVC = 6,
    H264_MVC = 7,
    HEVC = 8,
    VP8 = 9,
    VP9 = 10,
    AV1 = 11,
    NumCodecs = 12,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoSurfaceFormat {
    NV12 = 0,
    P016 = 1,
    YUV444 = 2,
    YUV444_16Bit = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoChromaFormat {
    Monochrome = 0,
    _420 = 1,
    _422 = 2,
    _444 = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoDeinterlaceMode {
    Weave = 0,
    Bob = 1,
    Adaptive = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum cudaVideoCreateFlags {
    /// Default (no flags).
    PreferCUVID = 0,
    /// Use dedicated hardware decoder (NVDEC).
    PreferDXVA = 1,
    /// Use CUDA-based decoder.
    PreferCUDA = 2,
}

// ─── Decoder creation params ─────────────────────────────────────────────

/// Cropping rectangle for decode output.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CUVIDDECODECREATEINFO_display_area {
    pub left: c_short,
    pub top: c_short,
    pub right: c_short,
    pub bottom: c_short,
}

/// Target output rectangle (scaling).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CUVIDDECODECREATEINFO_target_rect {
    pub left: c_short,
    pub top: c_short,
    pub right: c_short,
    pub bottom: c_short,
}

/// Parameters for `cuvidCreateDecoder`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDDECODECREATEINFO {
    pub ulWidth: c_ulong,
    pub ulHeight: c_ulong,
    pub ulNumDecodeSurfaces: c_ulong,
    pub CodecType: cudaVideoCodec,
    pub ChromaFormat: cudaVideoChromaFormat,
    pub ulCreationFlags: c_ulong,
    pub bitDepthMinus8: c_ulong,
    pub ulIntraDecodeOnly: c_ulong,
    pub ulMaxWidth: c_ulong,
    pub ulMaxHeight: c_ulong,
    pub Reserved1: c_ulong,
    pub display_area: CUVIDDECODECREATEINFO_display_area,
    pub OutputFormat: cudaVideoSurfaceFormat,
    pub DeinterlaceMode: cudaVideoDeinterlaceMode,
    pub ulTargetWidth: c_ulong,
    pub ulTargetHeight: c_ulong,
    pub ulNumOutputSurfaces: c_ulong,
    pub vidLock: *mut c_void,
    pub target_rect: CUVIDDECODECREATEINFO_target_rect,
    pub enableHistogram: c_ulong,
    pub Reserved2: [c_ulong; 4],
}

// ─── Picture params (decode a single frame) ──────────────────────────────

/// Simplified picture params — full struct is codec-union, we use opaque bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUVIDPICPARAMS {
    pub PicWidthInMbs: c_int,
    pub FrameHeightInMbs: c_int,
    pub CurrPicIdx: c_int,
    pub field_pic_flag: c_int,
    pub bottom_field_flag: c_int,
    pub second_field: c_int,
    pub nBitstreamDataLen: c_uint,
    pub pBitstreamData: *const u8,
    pub nNumSlices: c_uint,
    pub pSliceDataOffsets: *const c_uint,
    pub ref_pic_flag: c_int,
    pub intra_pic_flag: c_int,
    pub Reserved: [c_uint; 30],
    /// Codec-specific packed data (H.264/HEVC/VP9/AV1 union).
    pub CodecSpecific: [u8; 1024],
}

// ─── Parser callback types ───────────────────────────────────────────────

/// Video format information emitted by the parser.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDEOFORMAT {
    pub codec: cudaVideoCodec,
    pub frame_rate_num: c_uint,
    pub frame_rate_den: c_uint,
    pub progressive_sequence: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub min_num_decode_surfaces: u8,
    pub coded_width: c_uint,
    pub coded_height: c_uint,
    pub display_area_left: c_short,
    pub display_area_top: c_short,
    pub display_area_right: c_short,
    pub display_area_bottom: c_short,
    pub chroma_format: cudaVideoChromaFormat,
    pub bitrate: c_uint,
    pub display_aspect_ratio_x: c_uint,
    pub display_aspect_ratio_y: c_uint,
    pub video_signal_description: [u8; 8],
    pub seqhdr_data_length: c_uint,
}

/// Parser dispatch info for a decoded picture.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDPARSERDISPINFO {
    pub picture_index: c_int,
    pub progressive_frame: c_int,
    pub top_field_first: c_int,
    pub repeat_first_field: c_int,
    pub timestamp: c_ulonglong,
}

/// Callback: sequence header parsed (reports format).
pub type PFNVIDSEQUENCECALLBACK =
    unsafe extern "C" fn(user_data: *mut c_void, format: *mut CUVIDEOFORMAT) -> c_int;

/// Callback: a picture has been decoded.
pub type PFNVIDDECODECALLBACK =
    unsafe extern "C" fn(user_data: *mut c_void, pic_params: *mut CUVIDPICPARAMS) -> c_int;

/// Callback: a decoded picture is ready for display.
pub type PFNVIDDISPLAYCALLBACK =
    unsafe extern "C" fn(user_data: *mut c_void, disp_info: *mut CUVIDPARSERDISPINFO) -> c_int;

/// Parser creation params.
#[repr(C)]
pub struct CUVIDPARSERPARAMS {
    pub CodecType: cudaVideoCodec,
    pub ulMaxNumDecodeSurfaces: c_uint,
    pub ulClockRate: c_uint,
    pub ulErrorThreshold: c_uint,
    pub ulMaxDisplayDelay: c_uint,
    pub bAnnexb: c_uint,
    pub uReserved: c_uint,
    pub Reserved: [c_uint; 4],
    pub pUserData: *mut c_void,
    pub pfnSequenceCallback: Option<PFNVIDSEQUENCECALLBACK>,
    pub pfnDecodePicture: Option<PFNVIDDECODECALLBACK>,
    pub pfnDisplayPicture: Option<PFNVIDDISPLAYCALLBACK>,
    pub pvReserved2: [*mut c_void; 7],
    pub pExtVideoInfo: *mut c_void,
}

/// Bitstream packet fed to the parser.
#[repr(C)]
pub struct CUVIDSOURCEDATAPACKET {
    pub flags: c_ulong,
    pub payload_size: c_ulong,
    pub payload: *const u8,
    pub timestamp: c_ulonglong,
}

/// Parser input flags.
pub const CUVID_PKT_ENDOFSTREAM: c_ulong = 0x01;
pub const CUVID_PKT_TIMESTAMP: c_ulong = 0x02;
pub const CUVID_PKT_DISCONTINUITY: c_ulong = 0x04;

/// Processing params for `cuvidMapVideoFrame64`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUVIDPROCPARAMS {
    pub progressive_frame: c_int,
    pub second_field: c_int,
    pub top_field_first: c_int,
    pub unpaired_field: c_int,
    pub reserved_flags: c_uint,
    pub reserved_zero: c_uint,
    pub raw_input_dptr: c_ulonglong,
    pub raw_input_pitch: c_uint,
    pub raw_input_format: c_uint,
    pub raw_output_dptr: c_ulonglong,
    pub raw_output_pitch: c_uint,
    pub Reserved1: c_uint,
    pub output_stream: CUstream,
    pub Reserved: [c_uint; 46],
}

// ─── NVDEC functions ─────────────────────────────────────────────────────

unsafe extern "C" {
    pub fn cuvidCreateVideoParser(
        parser: *mut CUvideoparser,
        params: *mut CUVIDPARSERPARAMS,
    ) -> CUresult;

    pub fn cuvidParseVideoData(
        parser: CUvideoparser,
        packet: *mut CUVIDSOURCEDATAPACKET,
    ) -> CUresult;

    pub fn cuvidDestroyVideoParser(parser: CUvideoparser) -> CUresult;

    pub fn cuvidCreateDecoder(
        decoder: *mut CUvideodecoder,
        params: *mut CUVIDDECODECREATEINFO,
    ) -> CUresult;

    pub fn cuvidDecodePicture(decoder: CUvideodecoder, pic_params: *mut CUVIDPICPARAMS)
    -> CUresult;

    pub fn cuvidMapVideoFrame64(
        decoder: CUvideodecoder,
        pic_idx: c_int,
        dev_ptr: *mut CUdeviceptr,
        pitch: *mut c_uint,
        params: *mut CUVIDPROCPARAMS,
    ) -> CUresult;

    pub fn cuvidUnmapVideoFrame64(decoder: CUvideodecoder, dev_ptr: CUdeviceptr) -> CUresult;

    pub fn cuvidDestroyDecoder(decoder: CUvideodecoder) -> CUresult;
}

// ═══════════════════════════════════════════════════════════════════════════
//  NVENC — nvEncodeAPI.h
// ═══════════════════════════════════════════════════════════════════════════

/// NVENC status code.
pub type NVENCSTATUS = c_int;
pub const NV_ENC_SUCCESS: NVENCSTATUS = 0;
pub const NV_ENC_ERR_NO_ENCODE_DEVICE: NVENCSTATUS = 1;
pub const NV_ENC_ERR_UNSUPPORTED_DEVICE: NVENCSTATUS = 2;
pub const NV_ENC_ERR_INVALID_ENCODERDEVICE: NVENCSTATUS = 3;
pub const NV_ENC_ERR_INVALID_DEVICE: NVENCSTATUS = 4;
pub const NV_ENC_ERR_DEVICE_NOT_EXIST: NVENCSTATUS = 5;
pub const NV_ENC_ERR_INVALID_PTR: NVENCSTATUS = 6;
pub const NV_ENC_ERR_INVALID_EVENT: NVENCSTATUS = 7;
pub const NV_ENC_ERR_INVALID_PARAM: NVENCSTATUS = 8;
pub const NV_ENC_ERR_INVALID_CALL: NVENCSTATUS = 9;
pub const NV_ENC_ERR_OUT_OF_MEMORY: NVENCSTATUS = 10;
pub const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: NVENCSTATUS = 11;
pub const NV_ENC_ERR_UNSUPPORTED_PARAM: NVENCSTATUS = 12;
pub const NV_ENC_ERR_LOCK_BUSY: NVENCSTATUS = 13;
pub const NV_ENC_ERR_NOT_ENOUGH_BUFFER: NVENCSTATUS = 14;
pub const NV_ENC_ERR_INVALID_VERSION: NVENCSTATUS = 15;
pub const NV_ENC_ERR_MAP_FAILED: NVENCSTATUS = 16;
pub const NV_ENC_ERR_NEED_MORE_INPUT: NVENCSTATUS = 17;
pub const NV_ENC_ERR_ENCODER_BUSY: NVENCSTATUS = 18;
pub const NV_ENC_ERR_EVENT_NOT_REGISTERD: NVENCSTATUS = 19;
pub const NV_ENC_ERR_GENERIC: NVENCSTATUS = 20;

/// GUID type mirroring Windows GUID layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GUID {
    pub Data1: u32,
    pub Data2: u16,
    pub Data3: u16,
    pub Data4: [u8; 8],
}

// ─── Well-known GUIDs ────────────────────────────────────────────────────

/// H.264 encode GUID.
pub const NV_ENC_CODEC_H264_GUID: GUID = GUID {
    Data1: 0x6BC82762,
    Data2: 0x4E63,
    Data3: 0x4CA4,
    Data4: [0xAA, 0x85, 0x1A, 0x4F, 0x6A, 0x21, 0xF5, 0x07],
};

/// H.265/HEVC encode GUID.
pub const NV_ENC_CODEC_HEVC_GUID: GUID = GUID {
    Data1: 0x790CDC88,
    Data2: 0x4522,
    Data3: 0x4D7B,
    Data4: [0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03],
};

/// Low-latency high-quality preset GUID (P7).
pub const NV_ENC_PRESET_P7_GUID: GUID = GUID {
    Data1: 0x84848C12,
    Data2: 0x6F71,
    Data3: 0x4C13,
    Data4: [0x93, 0x1B, 0x53, 0xE5, 0x6F, 0x78, 0x84, 0x3B],
};

/// Balanced quality/performance preset GUID (P4).
pub const NV_ENC_PRESET_P4_GUID: GUID = GUID {
    Data1: 0x90A7B826,
    Data2: 0xDF06,
    Data3: 0x4862,
    Data4: [0xB9, 0xD2, 0xCD, 0x6D, 0x73, 0xA0, 0x86, 0x81],
};

/// H.265 Main profile GUID.
pub const NV_ENC_HEVC_PROFILE_MAIN_GUID: GUID = GUID {
    Data1: 0xB514C39A,
    Data2: 0xB55B,
    Data3: 0x40FA,
    Data4: [0x87, 0x87, 0x67, 0xED, 0x5E, 0x28, 0x49, 0x4D],
};

/// H.265 Main10 profile GUID.
pub const NV_ENC_HEVC_PROFILE_MAIN10_GUID: GUID = GUID {
    Data1: 0xFA4D2B6C,
    Data2: 0x3A5B,
    Data3: 0x411A,
    Data4: [0x80, 0x18, 0x0A, 0x3F, 0x5E, 0x3C, 0x9B, 0x44],
};

// ─── NVENC enums ─────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NV_ENC_DEVICE_TYPE {
    DIRECTX = 0,
    CUDA = 1,
    OPENGL = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NV_ENC_INPUT_RESOURCE_TYPE {
    DIRECTX = 0,
    CUDADEVICEPTR = 1,
    CUDAARRAY = 2,
    OPENGL_TEX = 3,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NV_ENC_BUFFER_FORMAT {
    UNDEFINED = 0x00000000,
    NV12 = 0x00000001,
    YV12 = 0x00000010,
    IYUV = 0x00000100,
    YUV444 = 0x00001000,
    YUV420_10BIT = 0x00010000,
    ARGB = 0x01000000,
    ABGR = 0x02000000,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NV_ENC_PIC_TYPE {
    P = 0,
    B = 1,
    I = 2,
    IDR = 3,
    BI = 4,
    SKIPPED = 5,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NV_ENC_PIC_STRUCT {
    FRAME = 0x01,
    FIELD_TOP_BOTTOM = 0x02,
    FIELD_BOTTOM_TOP = 0x03,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NV_ENC_TUNING_INFO {
    UNDEFINED = 0,
    HIGH_QUALITY = 1,
    LOW_LATENCY = 2,
    ULTRA_LOW_LATENCY = 3,
    LOSSLESS = 4,
}

// ─── NVENC params structs ────────────────────────────────────────────────

/// Matches `NVENCAPI_MAJOR_VERSION` in `nvEncodeAPI.h`.
pub const NVENCAPI_MAJOR_VERSION: u32 = 13;
/// Matches `NVENCAPI_MINOR_VERSION` in `nvEncodeAPI.h`.
pub const NVENCAPI_MINOR_VERSION: u32 = 0;
/// Matches `NVENCAPI_VERSION` in `nvEncodeAPI.h`.
pub const NVENCAPI_VERSION: u32 = NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24);

/// Matches `NVENCAPI_STRUCT_VERSION(ver)` in `nvEncodeAPI.h`.
#[inline]
pub const fn nvenc_struct_version(struct_ver: u32) -> u32 {
    NVENCAPI_VERSION | (struct_ver << 16) | (0x7 << 28)
}

pub const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 = nvenc_struct_version(1);
pub const NV_ENCODE_API_FUNCTION_LIST_VER: u32 = nvenc_struct_version(2);

/// Session open params.
#[repr(C)]
pub struct NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
    pub version: u32,
    pub deviceType: NV_ENC_DEVICE_TYPE,
    pub device: *mut c_void,
    pub reserved: *mut c_void,
    pub apiVersion: u32,
    pub reserved1: [u32; 253],
    pub reserved2: [*mut c_void; 64],
}

/// Rate control params (simplified).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NV_ENC_RC_PARAMS {
    pub version: u32,
    pub rateControlMode: u32,
    pub constQP_interP: u32,
    pub constQP_interB: u32,
    pub constQP_intra: u32,
    pub averageBitRate: u32,
    pub maxBitRate: u32,
    pub vbvBufferSize: u32,
    pub vbvInitialDelay: u32,
    pub reserved: [u32; 247],
}

/// HEVC-specific config (simplified).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NV_ENC_CONFIG_HEVC {
    pub level: u32,
    pub tier: u32,
    pub minCUSize: u32,
    pub maxCUSize: u32,
    pub reserved: [u32; 252],
}

/// Codec-specific config union (simplified — only HEVC used).
#[repr(C)]
#[derive(Clone, Copy)]
pub union NV_ENC_CODEC_CONFIG {
    pub hevcConfig: NV_ENC_CONFIG_HEVC,
    pub reserved: [u32; 256],
}

/// Encoder configuration.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NV_ENC_CONFIG {
    pub version: u32,
    pub profileGUID: GUID,
    pub gopLength: u32,
    pub frameIntervalP: i32,
    pub monoChromeEncoding: u32,
    pub frameFieldMode: u32,
    pub mvPrecision: u32,
    pub rcParams: NV_ENC_RC_PARAMS,
    pub encodeCodecConfig: NV_ENC_CODEC_CONFIG,
    pub reserved: [u32; 278],
    pub reserved2: [*mut c_void; 64],
}

/// Encoder initialization params.
#[repr(C)]
pub struct NV_ENC_INITIALIZE_PARAMS {
    pub version: u32,
    pub encodeGUID: GUID,
    pub presetGUID: GUID,
    pub encodeWidth: u32,
    pub encodeHeight: u32,
    pub darWidth: u32,
    pub darHeight: u32,
    pub frameRateNum: u32,
    pub frameRateDen: u32,
    pub enableEncodeAsync: u32,
    pub enablePTD: u32,
    pub reportSliceOffsets: u32,
    pub enableSubFrameWrite: u32,
    pub enableExternalMEHints: u32,
    pub enableMEOnlyMode: u32,
    pub enableWeightedPrediction: u32,
    pub enableOutputInVidmem: u32,
    pub reserved1: u32,
    pub privDataSize: u32,
    pub privData: *mut c_void,
    pub encodeConfig: *mut NV_ENC_CONFIG,
    pub maxEncodeWidth: u32,
    pub maxEncodeHeight: u32,
    pub maxMEHintCountsPerBlock: [u32; 2],
    pub tuningInfo: NV_ENC_TUNING_INFO,
    pub reserved: [u32; 289],
    pub reserved2: [*mut c_void; 64],
}

/// Register external resource.
#[repr(C)]
pub struct NV_ENC_REGISTER_RESOURCE {
    pub version: u32,
    pub resourceType: NV_ENC_INPUT_RESOURCE_TYPE,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub subResourceIndex: u32,
    pub resourceToRegister: *mut c_void,
    pub registeredResource: *mut c_void,
    pub bufferFormat: NV_ENC_BUFFER_FORMAT,
    pub bufferUsage: u32,
    pub pInputFencePoint: *mut c_void,
    pub pOutputFencePoint: *mut c_void,
    pub reserved: [u32; 247],
    pub reserved2: [*mut c_void; 62],
}

/// Map input resource.
#[repr(C)]
pub struct NV_ENC_MAP_INPUT_RESOURCE {
    pub version: u32,
    pub subResourceIndex: u32,
    pub inputResource: *mut c_void,
    pub registeredResource: *mut c_void,
    pub mappedResource: *mut c_void,
    pub mappedBufferFmt: NV_ENC_BUFFER_FORMAT,
    pub reserved: [u32; 251],
    pub reserved2: [*mut c_void; 63],
}

/// Create bitstream buffer.
#[repr(C)]
pub struct NV_ENC_CREATE_BITSTREAM_BUFFER {
    pub version: u32,
    pub bitstreamBuffer: *mut c_void,
    pub size: u32,
    pub memoryHeap: u32,
    pub reserved: [u32; 252],
    pub reserved2: [*mut c_void; 64],
}

/// Encode picture params.
#[repr(C)]
pub struct NV_ENC_PIC_PARAMS {
    pub version: u32,
    pub inputWidth: u32,
    pub inputHeight: u32,
    pub inputPitch: u32,
    pub encodePicFlags: u32,
    pub frameIdx: u32,
    pub inputTimeStamp: u64,
    pub inputDuration: u64,
    pub inputBuffer: *mut c_void,
    pub outputBitstream: *mut c_void,
    pub completionEvent: *mut c_void,
    pub bufferFmt: NV_ENC_BUFFER_FORMAT,
    pub pictureStruct: NV_ENC_PIC_STRUCT,
    pub pictureType: NV_ENC_PIC_TYPE,
    pub codecPicParams: [u32; 256],
    pub meHintCountsPerBlock: [u32; 2],
    pub meExternalHints: *mut c_void,
    pub reserved1: [u32; 6],
    pub reserved2: [*mut c_void; 2],
    pub qpDeltaMap: *mut i8,
    pub qpDeltaMapSize: u32,
    pub reservedBitFields: u32,
    pub meHintRefPicDist: [u32; 2],
    pub alphaBuffer: *mut c_void,
    pub reserved3: [u32; 286],
    pub reserved4: [*mut c_void; 60],
}

/// Lock bitstream output.
#[repr(C)]
pub struct NV_ENC_LOCK_BITSTREAM {
    pub version: u32,
    pub doNotWait: u32,
    pub ltrFrame: u32,
    pub reservedBitFields: u32,
    pub outputBitstream: *mut c_void,
    pub sliceOffsets: *mut u32,
    pub frameIdx: u32,
    pub hwEncodeStatus: u32,
    pub numSlices: u32,
    pub bitstreamSizeInBytes: u32,
    pub outputTimeStamp: u64,
    pub outputDuration: u64,
    pub bitstreamBufferPtr: *mut c_void,
    pub pictureType: NV_ENC_PIC_TYPE,
    pub pictureStruct: NV_ENC_PIC_STRUCT,
    pub frameAvgQP: u32,
    pub frameSatd: u32,
    pub ltrFrameIdx: u32,
    pub ltrFrameBitmap: u32,
    pub temporalId: u32,
    pub reserved: [u32; 13],
    pub intraMBCount: u32,
    pub interMBCount: u32,
    pub averageMVX: i32,
    pub averageMVY: i32,
    pub reserved1: [u32; 226],
    pub reserved2: [*mut c_void; 64],
}

/// End-of-stream flag for NV_ENC_PIC_PARAMS.
pub const NV_ENC_PIC_FLAG_EOS: u32 = 0x01;
/// Force IDR flag.
pub const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 0x04;

// ─── NVENC function pointer table ────────────────────────────────────────

/// Subset of the NV_ENCODE_API_FUNCTION_LIST that we actually call.
#[repr(C)]
pub struct NV_ENCODE_API_FUNCTION_LIST {
    pub version: u32,
    pub reserved: u32,
    pub nvEncOpenEncodeSession: *const c_void,
    pub nvEncGetEncodeGUIDCount: *const c_void,
    pub nvEncGetEncodeProfileGUIDCount: *const c_void,
    pub nvEncGetEncodeProfileGUIDs: *const c_void,
    pub nvEncGetEncodeGUIDs: *const c_void,
    pub nvEncGetInputFormatCount: *const c_void,
    pub nvEncGetInputFormats: *const c_void,
    pub nvEncGetEncodeCaps: *const c_void,
    pub nvEncGetEncodePresetCount: *const c_void,
    pub nvEncGetEncodePresetGUIDs: *const c_void,
    pub nvEncGetEncodePresetConfig: *const c_void,
    pub nvEncInitializeEncoder:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_INITIALIZE_PARAMS) -> NVENCSTATUS>,
    pub nvEncCreateInputBuffer: *const c_void,
    pub nvEncDestroyInputBuffer: *const c_void,
    pub nvEncCreateBitstreamBuffer: Option<
        unsafe extern "C" fn(*mut c_void, *mut NV_ENC_CREATE_BITSTREAM_BUFFER) -> NVENCSTATUS,
    >,
    pub nvEncDestroyBitstreamBuffer:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    pub nvEncEncodePicture:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_PIC_PARAMS) -> NVENCSTATUS>,
    pub nvEncLockBitstream:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_LOCK_BITSTREAM) -> NVENCSTATUS>,
    pub nvEncUnlockBitstream: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    pub nvEncLockInputBuffer: *const c_void,
    pub nvEncUnlockInputBuffer: *const c_void,
    pub nvEncGetEncodeStats: *const c_void,
    pub nvEncGetSequenceParams: *const c_void,
    pub nvEncRegisterAsyncEvent: *const c_void,
    pub nvEncUnregisterAsyncEvent: *const c_void,
    pub nvEncMapInputResource:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_MAP_INPUT_RESOURCE) -> NVENCSTATUS>,
    pub nvEncUnmapInputResource:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    pub nvEncDestroyEncoder: Option<unsafe extern "C" fn(*mut c_void) -> NVENCSTATUS>,
    pub nvEncInvalidateRefFrames: *const c_void,
    pub nvEncOpenEncodeSessionEx: Option<
        unsafe extern "C" fn(
            *mut NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
            *mut *mut c_void,
        ) -> NVENCSTATUS,
    >,
    pub nvEncRegisterResource:
        Option<unsafe extern "C" fn(*mut c_void, *mut NV_ENC_REGISTER_RESOURCE) -> NVENCSTATUS>,
    pub nvEncUnregisterResource:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NVENCSTATUS>,
    pub nvEncReconfigureEncoder: *const c_void,
    pub reserved1: *const c_void,
    pub nvEncCreateMVBuffer: *const c_void,
    pub nvEncDestroyMVBuffer: *const c_void,
    pub nvEncRunMotionEstimationOnly: *const c_void,
    pub nvEncGetLastErrorString: *const c_void,
    pub nvEncSetIOCudaStreams: *const c_void,
    pub nvEncGetEncodePresetConfigEx: Option<
        unsafe extern "C" fn(
            *mut c_void,
            GUID,
            GUID,
            NV_ENC_TUNING_INFO,
            *mut NV_ENC_PRESET_CONFIG,
        ) -> NVENCSTATUS,
    >,
    pub nvEncGetSequenceParamEx: *const c_void,
    pub nvEncRestoreEncoderState: *const c_void,
    pub nvEncLookaheadPicture: *const c_void,
    pub reserved2: [*const c_void; 275],
}

/// Preset config wrapper.
#[repr(C)]
pub struct NV_ENC_PRESET_CONFIG {
    pub version: u32,
    pub presetCfg: NV_ENC_CONFIG,
    pub reserved: [u32; 255],
    pub reserved2: [*mut c_void; 64],
}

unsafe extern "C" {
    pub fn NvEncodeAPIGetMaxSupportedVersion(version: *mut u32) -> NVENCSTATUS;
    /// Entry point to get the NVENC function table.
    pub fn NvEncodeAPICreateInstance(
        function_list: *mut NV_ENCODE_API_FUNCTION_LIST,
    ) -> NVENCSTATUS;
}

// ═══════════════════════════════════════════════════════════════════════════
//  CUDA DRIVER — event functions (not in cudarc)
// ═══════════════════════════════════════════════════════════════════════════

/// CUDA event handle.
pub type CUevent = *mut c_void;

pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x02;

unsafe extern "C" {
    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuEventCreate(phEvent: *mut CUevent, Flags: c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;
    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
    pub fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: c_uint) -> CUresult;
    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
}

// ═══════════════════════════════════════════════════════════════════════════
//  CUDA DRIVER — async memcpy (for D2D copy of decoded surfaces)
// ═══════════════════════════════════════════════════════════════════════════

unsafe extern "C" {
    pub fn cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult;
}

/// 2D memory copy descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUDA_MEMCPY2D {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: *const c_void,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: *mut c_void,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CUmemorytype {
    Host = 0x01,
    Device = 0x02,
    Array = 0x03,
    Unified = 0x04,
}

// ═══════════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a CUDA result to an engine Result.
#[inline]
pub fn check_cu(result: CUresult, context: &str) -> rave_core::error::Result<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(rave_core::error::EngineError::Decode(format!(
            "{context}: CUDA error code {result}"
        )))
    }
}

/// Convert an NVENC status to an engine Result.
#[inline]
pub fn check_nvenc(status: NVENCSTATUS, context: &str) -> rave_core::error::Result<()> {
    if status == NV_ENC_SUCCESS {
        Ok(())
    } else {
        Err(rave_core::error::EngineError::Encode(format!(
            "{context}: NVENC error code {status} ({})",
            nvenc_status_name(status)
        )))
    }
}

/// Human-readable NVENC status names for diagnostics.
#[inline]
pub const fn nvenc_status_name(status: NVENCSTATUS) -> &'static str {
    match status {
        NV_ENC_SUCCESS => "NV_ENC_SUCCESS",
        NV_ENC_ERR_NO_ENCODE_DEVICE => "NV_ENC_ERR_NO_ENCODE_DEVICE",
        NV_ENC_ERR_UNSUPPORTED_DEVICE => "NV_ENC_ERR_UNSUPPORTED_DEVICE",
        NV_ENC_ERR_INVALID_ENCODERDEVICE => "NV_ENC_ERR_INVALID_ENCODERDEVICE",
        NV_ENC_ERR_INVALID_DEVICE => "NV_ENC_ERR_INVALID_DEVICE",
        NV_ENC_ERR_DEVICE_NOT_EXIST => "NV_ENC_ERR_DEVICE_NOT_EXIST",
        NV_ENC_ERR_INVALID_PTR => "NV_ENC_ERR_INVALID_PTR",
        NV_ENC_ERR_INVALID_EVENT => "NV_ENC_ERR_INVALID_EVENT",
        NV_ENC_ERR_INVALID_PARAM => "NV_ENC_ERR_INVALID_PARAM",
        NV_ENC_ERR_INVALID_CALL => "NV_ENC_ERR_INVALID_CALL",
        NV_ENC_ERR_OUT_OF_MEMORY => "NV_ENC_ERR_OUT_OF_MEMORY",
        NV_ENC_ERR_ENCODER_NOT_INITIALIZED => "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
        NV_ENC_ERR_UNSUPPORTED_PARAM => "NV_ENC_ERR_UNSUPPORTED_PARAM",
        NV_ENC_ERR_LOCK_BUSY => "NV_ENC_ERR_LOCK_BUSY",
        NV_ENC_ERR_NOT_ENOUGH_BUFFER => "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
        NV_ENC_ERR_INVALID_VERSION => "NV_ENC_ERR_INVALID_VERSION",
        NV_ENC_ERR_MAP_FAILED => "NV_ENC_ERR_MAP_FAILED",
        NV_ENC_ERR_NEED_MORE_INPUT => "NV_ENC_ERR_NEED_MORE_INPUT",
        NV_ENC_ERR_ENCODER_BUSY => "NV_ENC_ERR_ENCODER_BUSY",
        NV_ENC_ERR_EVENT_NOT_REGISTERD => "NV_ENC_ERR_EVENT_NOT_REGISTERD",
        NV_ENC_ERR_GENERIC => "NV_ENC_ERR_GENERIC",
        _ => "NV_ENC_ERR_UNKNOWN",
    }
}
