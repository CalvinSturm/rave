//! FFmpeg-based container demuxer — [`BitstreamSource`] impl for MP4/MKV/MOV.
//!
//! Reads compressed video packets from a container file and converts them
//! from MP4 length-prefixed format to Annex B NAL units via the appropriate
//! bitstream filter (`h264_mp4toannexb` or `hevc_mp4toannexb`).

use std::ptr;

use ffmpeg_sys_next::*;

/// POSIX EAGAIN — used with AVERROR() for "try again" semantics.
const EAGAIN: i32 = 11;

use crate::ffmpeg_sys::{
    // BSF FFI (missing from ffmpeg-sys-next v8)
    AVBSFContext,
    av_bsf_alloc,
    av_bsf_free,
    av_bsf_get_by_name,
    av_bsf_init,
    av_bsf_receive_packet,
    av_bsf_send_packet,
    check_ffmpeg,
    to_cstring,
};
use rave_core::codec_traits::{BitstreamPacket, BitstreamSource};
use rave_core::error::{EngineError, Result};
use rave_core::ffi_types::cudaVideoCodec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PacketSlot {
    Read,
    Pending,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SendInput<P> {
    Packet(P),
    Flush,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SendOutcome<P> {
    Accepted,
    Again(SendInput<P>),
}

enum RecvOutcome {
    Packet(BitstreamPacket),
    Again,
    Eof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadOutcome<P> {
    Packet(P),
    Eof,
}

trait BsfIo {
    type Packet: Copy + Eq;

    fn recv_filtered(&mut self) -> Result<RecvOutcome>;
    fn send_input(&mut self, input: SendInput<Self::Packet>) -> Result<SendOutcome<Self::Packet>>;
    fn read_next_video_packet(&mut self) -> Result<ReadOutcome<Self::Packet>>;
}

#[derive(Debug)]
struct BsfMachine<P> {
    pending: Option<SendInput<P>>,
    flushing: bool,
    flush_sent: bool,
    terminal_eos: bool,
    idle_loops: usize,
}

impl<P> Default for BsfMachine<P> {
    fn default() -> Self {
        Self {
            pending: None,
            flushing: false,
            flush_sent: false,
            terminal_eos: false,
            idle_loops: 0,
        }
    }
}

impl<P: Copy + Eq> BsfMachine<P> {
    fn poll<IO>(&mut self, io: &mut IO) -> Result<Option<BitstreamPacket>>
    where
        IO: BsfIo<Packet = P>,
    {
        const MAX_IDLE_LOOPS: usize = 1024;

        if self.terminal_eos {
            return Ok(None);
        }

        loop {
            let mut progressed = false;

            match io.recv_filtered()? {
                RecvOutcome::Packet(pkt) => return Ok(Some(pkt)),
                RecvOutcome::Again => {}
                RecvOutcome::Eof => {
                    if self.flush_sent {
                        self.terminal_eos = true;
                        return Ok(None);
                    }
                    if !self.flushing {
                        return Err(EngineError::Demux(
                            "Bitstream filter reached EOF before flush was initiated".into(),
                        ));
                    }
                }
            }

            if let Some(input) = self.pending.take() {
                match io.send_input(input)? {
                    SendOutcome::Accepted => {
                        progressed = true;
                    }
                    SendOutcome::Again(input) => {
                        self.pending = Some(input);
                    }
                }
            } else if self.flushing {
                if !self.flush_sent {
                    match io.send_input(SendInput::Flush)? {
                        SendOutcome::Accepted => {
                            self.flush_sent = true;
                            progressed = true;
                        }
                        SendOutcome::Again(input) => {
                            self.pending = Some(input);
                        }
                    }
                }
            } else {
                match io.read_next_video_packet()? {
                    ReadOutcome::Packet(pkt) => {
                        progressed = true;
                        match io.send_input(SendInput::Packet(pkt))? {
                            SendOutcome::Accepted => {}
                            SendOutcome::Again(input) => {
                                self.pending = Some(input);
                            }
                        }
                    }
                    ReadOutcome::Eof => {
                        self.flushing = true;
                        progressed = true;
                    }
                }
            }

            if !progressed {
                self.idle_loops += 1;
                if self.idle_loops > MAX_IDLE_LOOPS {
                    return Err(EngineError::Demux(
                        "Bitstream filter state machine stalled (no forward progress)".into(),
                    ));
                }
            } else {
                self.idle_loops = 0;
            }
        }
    }
}

/// Demuxes a container file and produces Annex B bitstream packets.
pub struct FfmpegDemuxer {
    fmt_ctx: *mut AVFormatContext,
    bsf_ctx: *mut AVBSFContext,
    video_stream_index: i32,
    /// Packet for reading from the container.
    pkt_read: *mut AVPacket,
    /// Packet for receiving filtered output from BSF.
    pkt_filtered: *mut AVPacket,
    /// Packet retained when `av_bsf_send_packet` returns EAGAIN.
    pkt_pending: *mut AVPacket,
    /// Stream time_base for PTS rescaling to microseconds.
    time_base: AVRational,
    eos: bool,
    bsf_machine: BsfMachine<PacketSlot>,
}

// SAFETY: All FFmpeg operations happen on a single thread (the decode
// blocking task). The raw pointers are not shared across threads.
unsafe impl Send for FfmpegDemuxer {}

impl FfmpegDemuxer {
    /// Open a container and prepare the Annex B bitstream filter.
    pub fn new(path: &std::path::Path, codec: cudaVideoCodec) -> Result<Self> {
        let path_str = path
            .to_str()
            .ok_or_else(|| EngineError::Demux("Non-UTF8 path".into()))?;
        let c_path = to_cstring(path_str).map_err(EngineError::Demux)?;

        // ── Open container ──
        let mut fmt_ctx: *mut AVFormatContext = ptr::null_mut();
        let ret = unsafe {
            avformat_open_input(&mut fmt_ctx, c_path.as_ptr(), ptr::null(), ptr::null_mut())
        };
        check_ffmpeg(ret, "avformat_open_input").map_err(|e| EngineError::Demux(e.to_string()))?;

        let ret = unsafe { avformat_find_stream_info(fmt_ctx, ptr::null_mut()) };
        if ret < 0 {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            check_ffmpeg(ret, "avformat_find_stream_info")
                .map_err(|e| EngineError::Demux(e.to_string()))?;
        }

        // ── Find video stream ──
        let stream_index = unsafe {
            av_find_best_stream(
                fmt_ctx,
                AVMediaType::AVMEDIA_TYPE_VIDEO,
                -1,
                -1,
                ptr::null_mut(),
                0,
            )
        };
        if stream_index < 0 {
            unsafe { avformat_close_input(&mut fmt_ctx) };
            return Err(EngineError::Demux(
                "No video stream found in container".into(),
            ));
        }

        let stream = unsafe { &*(*(*fmt_ctx).streams.add(stream_index as usize)) };
        let time_base = stream.time_base;

        // ── Initialize bitstream filter (MP4 → Annex B) ──
        let bsf_name = match codec {
            cudaVideoCodec::H264 => Some(c"h264_mp4toannexb"),
            cudaVideoCodec::HEVC => Some(c"hevc_mp4toannexb"),
            _ => None,
        };

        let mut bsf_ctx: *mut AVBSFContext = ptr::null_mut();
        if let Some(bsf_name) = bsf_name {
            let bsf = unsafe { av_bsf_get_by_name(bsf_name.as_ptr()) };
            if bsf.is_null() {
                unsafe { avformat_close_input(&mut fmt_ctx) };
                return Err(EngineError::BitstreamFilter(format!(
                    "BSF {:?} not found — FFmpeg build may be incomplete",
                    bsf_name
                )));
            }

            let ret = unsafe { av_bsf_alloc(bsf, &mut bsf_ctx) };
            if ret < 0 {
                unsafe { avformat_close_input(&mut fmt_ctx) };
                check_ffmpeg(ret, "av_bsf_alloc").map_err(|e| EngineError::Demux(e.to_string()))?;
            }

            // Copy codec parameters from the stream to the BSF.
            let ret = unsafe { avcodec_parameters_copy((*bsf_ctx).par_in, stream.codecpar) };
            if ret < 0 {
                unsafe {
                    av_bsf_free(&mut bsf_ctx);
                    avformat_close_input(&mut fmt_ctx);
                }
                check_ffmpeg(ret, "avcodec_parameters_copy")
                    .map_err(|e| EngineError::Demux(e.to_string()))?;
            }

            let ret = unsafe { av_bsf_init(bsf_ctx) };
            if ret < 0 {
                unsafe {
                    av_bsf_free(&mut bsf_ctx);
                    avformat_close_input(&mut fmt_ctx);
                }
                check_ffmpeg(ret, "av_bsf_init").map_err(|e| EngineError::Demux(e.to_string()))?;
            }
        }

        // ── Allocate packets ──
        let mut pkt_read = unsafe { av_packet_alloc() };
        let mut pkt_filtered = unsafe { av_packet_alloc() };
        let mut pkt_pending = unsafe { av_packet_alloc() };

        if pkt_read.is_null() || pkt_filtered.is_null() || pkt_pending.is_null() {
            unsafe {
                if !pkt_pending.is_null() {
                    av_packet_free(&mut pkt_pending);
                }
                if !pkt_filtered.is_null() {
                    av_packet_free(&mut pkt_filtered);
                }
                if !pkt_read.is_null() {
                    av_packet_free(&mut pkt_read);
                }
                if !bsf_ctx.is_null() {
                    av_bsf_free(&mut bsf_ctx);
                }
                avformat_close_input(&mut fmt_ctx);
            }
            return Err(EngineError::Demux("Failed to allocate AVPacket".into()));
        }

        if bsf_ctx.is_null() {
            tracing::info!(
                path = %path.display(),
                ?codec,
                stream_index,
                "FFmpeg demuxer opened (no BSF needed)"
            );
        } else {
            tracing::info!(
                path = %path.display(),
                ?codec,
                stream_index,
                "FFmpeg demuxer opened with Annex B BSF"
            );
        }

        Ok(Self {
            fmt_ctx,
            bsf_ctx,
            video_stream_index: stream_index,
            pkt_read,
            pkt_filtered,
            pkt_pending,
            time_base,
            eos: false,
            bsf_machine: BsfMachine::default(),
        })
    }

    /// Rescale PTS from stream time_base to microseconds.
    fn rescale_pts(&self, pts: i64) -> i64 {
        if pts == AV_NOPTS_VALUE {
            return 0;
        }
        let us_tb = AVRational {
            num: 1,
            den: 1_000_000,
        };
        unsafe { av_rescale_q(pts, self.time_base, us_tb) }
    }

    fn copy_packet_data(pkt: &AVPacket) -> Result<Vec<u8>> {
        if pkt.size <= 0 {
            return Ok(Vec::new());
        }
        if pkt.data.is_null() {
            return Err(EngineError::Demux(
                "FFmpeg produced packet with null data pointer".into(),
            ));
        }
        // SAFETY: `pkt.data` is valid for `pkt.size` bytes when size > 0.
        Ok(unsafe { std::slice::from_raw_parts(pkt.data, pkt.size as usize) }.to_vec())
    }

    fn bsf_recv_filtered(&mut self) -> Result<RecvOutcome> {
        let ret = unsafe { av_bsf_receive_packet(self.bsf_ctx, self.pkt_filtered) };
        if ret == 0 {
            let pkt = unsafe { &*self.pkt_filtered };
            let data = Self::copy_packet_data(pkt)?;
            let pts = self.rescale_pts(pkt.pts);
            let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
            unsafe { av_packet_unref(self.pkt_filtered) };
            if data.is_empty() {
                tracing::debug!("Skipping empty demuxed packet after BSF");
                return Ok(RecvOutcome::Again);
            }
            return Ok(RecvOutcome::Packet(BitstreamPacket {
                data,
                pts,
                is_keyframe,
            }));
        }
        if ret == AVERROR(EAGAIN) {
            return Ok(RecvOutcome::Again);
        }
        if ret == AVERROR_EOF {
            return Ok(RecvOutcome::Eof);
        }
        check_ffmpeg(ret, "av_bsf_receive_packet")
            .map_err(|e| EngineError::Demux(e.to_string()))?;
        Err(EngineError::Demux(
            "unreachable: av_bsf_receive_packet error should have returned".into(),
        ))
    }

    fn bsf_send_input(&mut self, input: SendInput<PacketSlot>) -> Result<SendOutcome<PacketSlot>> {
        match input {
            SendInput::Packet(PacketSlot::Read) => {
                let ret = unsafe { av_bsf_send_packet(self.bsf_ctx, self.pkt_read) };
                if ret == 0 {
                    unsafe { av_packet_unref(self.pkt_read) };
                    return Ok(SendOutcome::Accepted);
                }
                if ret == AVERROR(EAGAIN) {
                    unsafe { av_packet_move_ref(self.pkt_pending, self.pkt_read) };
                    return Ok(SendOutcome::Again(SendInput::Packet(PacketSlot::Pending)));
                }
                unsafe { av_packet_unref(self.pkt_read) };
                check_ffmpeg(ret, "av_bsf_send_packet")
                    .map_err(|e| EngineError::Demux(e.to_string()))?;
                Err(EngineError::Demux(
                    "unreachable: av_bsf_send_packet error should have returned".into(),
                ))
            }
            SendInput::Packet(PacketSlot::Pending) => {
                let ret = unsafe { av_bsf_send_packet(self.bsf_ctx, self.pkt_pending) };
                if ret == 0 {
                    unsafe { av_packet_unref(self.pkt_pending) };
                    return Ok(SendOutcome::Accepted);
                }
                if ret == AVERROR(EAGAIN) {
                    return Ok(SendOutcome::Again(SendInput::Packet(PacketSlot::Pending)));
                }
                unsafe { av_packet_unref(self.pkt_pending) };
                check_ffmpeg(ret, "av_bsf_send_packet (pending)")
                    .map_err(|e| EngineError::Demux(e.to_string()))?;
                Err(EngineError::Demux(
                    "unreachable: av_bsf_send_packet pending error should have returned".into(),
                ))
            }
            SendInput::Flush => {
                let ret = unsafe { av_bsf_send_packet(self.bsf_ctx, ptr::null()) };
                if ret == 0 {
                    return Ok(SendOutcome::Accepted);
                }
                if ret == AVERROR(EAGAIN) {
                    return Ok(SendOutcome::Again(SendInput::Flush));
                }
                check_ffmpeg(ret, "av_bsf_send_packet (flush)")
                    .map_err(|e| EngineError::Demux(e.to_string()))?;
                Err(EngineError::Demux(
                    "unreachable: av_bsf_send_packet flush error should have returned".into(),
                ))
            }
        }
    }

    fn bsf_read_next_video_packet(&mut self) -> Result<ReadOutcome<PacketSlot>> {
        loop {
            let ret = unsafe { av_read_frame(self.fmt_ctx, self.pkt_read) };
            if ret < 0 {
                if ret == AVERROR_EOF {
                    return Ok(ReadOutcome::Eof);
                }
                check_ffmpeg(ret, "av_read_frame")
                    .map_err(|e| EngineError::Demux(e.to_string()))?;
                return Err(EngineError::Demux(
                    "unreachable: av_read_frame error should have returned".into(),
                ));
            }

            let pkt = unsafe { &*self.pkt_read };
            if pkt.stream_index != self.video_stream_index {
                unsafe { av_packet_unref(self.pkt_read) };
                continue;
            }

            return Ok(ReadOutcome::Packet(PacketSlot::Read));
        }
    }

    fn read_packet_passthrough(&mut self) -> Result<Option<BitstreamPacket>> {
        loop {
            let ret = unsafe { av_read_frame(self.fmt_ctx, self.pkt_read) };
            if ret < 0 {
                if ret == AVERROR_EOF {
                    self.eos = true;
                    return Ok(None);
                }
                check_ffmpeg(ret, "av_read_frame")
                    .map_err(|e| EngineError::Demux(e.to_string()))?;
                return Err(EngineError::Demux(
                    "unreachable: av_read_frame error should have returned".into(),
                ));
            }

            let pkt = unsafe { &*self.pkt_read };
            if pkt.stream_index != self.video_stream_index {
                unsafe { av_packet_unref(self.pkt_read) };
                continue;
            }

            let data = Self::copy_packet_data(pkt)?;
            let pts = self.rescale_pts(pkt.pts);
            let is_keyframe = (pkt.flags & AV_PKT_FLAG_KEY) != 0;
            unsafe { av_packet_unref(self.pkt_read) };
            if data.is_empty() {
                tracing::debug!("Skipping empty demuxed packet (no BSF)");
                continue;
            }
            return Ok(Some(BitstreamPacket {
                data,
                pts,
                is_keyframe,
            }));
        }
    }
}

struct FfmpegBsfIo<'a> {
    demuxer: &'a mut FfmpegDemuxer,
}

impl BsfIo for FfmpegBsfIo<'_> {
    type Packet = PacketSlot;

    fn recv_filtered(&mut self) -> Result<RecvOutcome> {
        self.demuxer.bsf_recv_filtered()
    }

    fn send_input(&mut self, input: SendInput<Self::Packet>) -> Result<SendOutcome<Self::Packet>> {
        self.demuxer.bsf_send_input(input)
    }

    fn read_next_video_packet(&mut self) -> Result<ReadOutcome<Self::Packet>> {
        self.demuxer.bsf_read_next_video_packet()
    }
}

impl BitstreamSource for FfmpegDemuxer {
    fn read_packet(&mut self) -> Result<Option<BitstreamPacket>> {
        if self.eos {
            return Ok(None);
        }

        if self.bsf_ctx.is_null() {
            return self.read_packet_passthrough();
        }

        let mut machine = std::mem::take(&mut self.bsf_machine);
        let mut io = FfmpegBsfIo { demuxer: self };
        let out = machine.poll(&mut io);
        self.bsf_machine = machine;

        if matches!(out, Ok(None)) {
            self.eos = true;
        }

        out
    }
}

impl Drop for FfmpegDemuxer {
    fn drop(&mut self) {
        // Free in reverse allocation order.
        unsafe {
            av_packet_free(&mut self.pkt_pending);
            av_packet_free(&mut self.pkt_filtered);
            av_packet_free(&mut self.pkt_read);
            if !self.bsf_ctx.is_null() {
                av_bsf_free(&mut self.bsf_ctx);
            }
            if !self.fmt_ctx.is_null() {
                avformat_close_input(&mut self.fmt_ctx);
            }
        }
        tracing::debug!("FFmpeg demuxer destroyed");
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};

    use super::*;

    struct DelayedPacket {
        pkt: BitstreamPacket,
        again_before_recv: usize,
    }

    struct FakeIo {
        reads: VecDeque<ReadOutcome<u32>>,
        recv_queue: VecDeque<DelayedPacket>,
        flush_tail: VecDeque<DelayedPacket>,
        eagain_before_accept: HashMap<u32, usize>,
        send_attempts: HashMap<u32, usize>,
        outputs_per_packet: HashMap<u32, Vec<DelayedPacket>>,
        flush_sent: bool,
        send_log: Vec<SendInput<u32>>,
    }

    impl FakeIo {
        fn new(reads: Vec<ReadOutcome<u32>>) -> Self {
            Self {
                reads: reads.into(),
                recv_queue: VecDeque::new(),
                flush_tail: VecDeque::new(),
                eagain_before_accept: HashMap::new(),
                send_attempts: HashMap::new(),
                outputs_per_packet: HashMap::new(),
                flush_sent: false,
                send_log: Vec::new(),
            }
        }

        fn push_initial_delayed(&mut self, packet: BitstreamPacket, again_before_recv: usize) {
            self.recv_queue.push_back(DelayedPacket {
                pkt: packet,
                again_before_recv,
            });
        }

        fn with_initial_recv(mut self, packets: Vec<BitstreamPacket>) -> Self {
            for packet in packets {
                self.push_initial_delayed(packet, 0);
            }
            self
        }

        fn push_flush_tail_delayed(&mut self, packet: BitstreamPacket, again_before_recv: usize) {
            self.flush_tail.push_back(DelayedPacket {
                pkt: packet,
                again_before_recv,
            });
        }

        fn with_flush_tail(mut self, packets: Vec<BitstreamPacket>) -> Self {
            for packet in packets {
                self.push_flush_tail_delayed(packet, 0);
            }
            self
        }

        fn with_eagain_rule(mut self, packet_id: u32, eagain_count: usize) -> Self {
            self.eagain_before_accept.insert(packet_id, eagain_count);
            self
        }

        fn push_output_delayed(
            &mut self,
            packet_id: u32,
            packet: BitstreamPacket,
            again_before_recv: usize,
        ) {
            self.outputs_per_packet
                .entry(packet_id)
                .or_default()
                .push(DelayedPacket {
                    pkt: packet,
                    again_before_recv,
                });
        }

        fn with_outputs(mut self, packet_id: u32, outputs: Vec<BitstreamPacket>) -> Self {
            for packet in outputs {
                self.push_output_delayed(packet_id, packet, 0);
            }
            self
        }
    }

    impl BsfIo for FakeIo {
        type Packet = u32;

        fn recv_filtered(&mut self) -> Result<RecvOutcome> {
            if let Some(front) = self.recv_queue.front_mut() {
                if front.again_before_recv > 0 {
                    front.again_before_recv -= 1;
                    return Ok(RecvOutcome::Again);
                }
                let packet = self
                    .recv_queue
                    .pop_front()
                    .expect("front packet should exist after delay is consumed");
                return Ok(RecvOutcome::Packet(packet.pkt));
            }
            if self.flush_sent {
                return Ok(RecvOutcome::Eof);
            }
            Ok(RecvOutcome::Again)
        }

        fn send_input(
            &mut self,
            input: SendInput<Self::Packet>,
        ) -> Result<SendOutcome<Self::Packet>> {
            self.send_log.push(input);
            match input {
                SendInput::Packet(id) => {
                    let attempts = self.send_attempts.entry(id).or_insert(0);
                    *attempts += 1;
                    let eagain_count = self.eagain_before_accept.get(&id).copied().unwrap_or(0);
                    if *attempts <= eagain_count {
                        return Ok(SendOutcome::Again(SendInput::Packet(id)));
                    }

                    if let Some(outputs) = self.outputs_per_packet.remove(&id) {
                        self.recv_queue.extend(outputs);
                    }
                    Ok(SendOutcome::Accepted)
                }
                SendInput::Flush => {
                    self.flush_sent = true;
                    self.recv_queue.extend(self.flush_tail.drain(..));
                    Ok(SendOutcome::Accepted)
                }
            }
        }

        fn read_next_video_packet(&mut self) -> Result<ReadOutcome<Self::Packet>> {
            Ok(self.reads.pop_front().unwrap_or(ReadOutcome::Eof))
        }
    }

    fn pkt(id: u8) -> BitstreamPacket {
        BitstreamPacket {
            data: vec![id],
            pts: id as i64,
            is_keyframe: id.is_multiple_of(2),
        }
    }

    #[test]
    fn empty_packet_data_is_safe() {
        let pkt: AVPacket = unsafe { std::mem::zeroed() };
        let data = FfmpegDemuxer::copy_packet_data(&pkt).expect("empty packet should be accepted");
        assert!(data.is_empty());
    }

    #[test]
    fn flush_drains_all_filtered_packets_before_none() {
        let mut io = FakeIo::new(vec![ReadOutcome::Packet(1), ReadOutcome::Eof])
            .with_outputs(1, vec![pkt(10)])
            .with_flush_tail(vec![pkt(11), pkt(12)]);

        let mut machine = BsfMachine::<u32>::default();
        let mut out = Vec::new();
        while let Some(pkt) = machine.poll(&mut io).expect("poll") {
            out.push(pkt.data[0]);
        }

        assert_eq!(out, vec![10, 11, 12]);
        assert_eq!(
            io.send_log,
            vec![SendInput::Packet(1), SendInput::Flush],
            "flush should be sent exactly once after EOF"
        );
    }

    #[test]
    fn eagain_retries_same_packet_before_reading_next() {
        let mut io = FakeIo::new(vec![
            ReadOutcome::Packet(1),
            ReadOutcome::Packet(2),
            ReadOutcome::Eof,
        ])
        .with_initial_recv(vec![pkt(42)])
        .with_eagain_rule(1, 1)
        .with_outputs(1, vec![pkt(10)])
        .with_outputs(2, vec![pkt(20)]);

        let mut machine = BsfMachine::<u32>::default();
        let mut out = Vec::new();
        while let Some(pkt) = machine.poll(&mut io).expect("poll") {
            out.push(pkt.data[0]);
        }

        assert_eq!(out, vec![42, 10, 20]);
        assert_eq!(
            io.send_log,
            vec![
                SendInput::Packet(1),
                SendInput::Packet(1),
                SendInput::Packet(2),
                SendInput::Flush,
            ],
            "packet 1 must be retried after EAGAIN before packet 2 is read"
        );
    }

    fn retry_vectors(len: usize, max_retry: usize) -> Vec<Vec<usize>> {
        if len == 0 {
            return vec![Vec::new()];
        }
        let tails = retry_vectors(len - 1, max_retry);
        let mut out = Vec::new();
        for head in 0..=max_retry {
            for tail in &tails {
                let mut seq = Vec::with_capacity(len);
                seq.push(head);
                seq.extend(tail.iter().copied());
                out.push(seq);
            }
        }
        out
    }

    #[test]
    fn bsf_machine_permutations_preserve_order_and_eos_semantics() {
        let max_retry = 2usize;
        let mut scenario_count = 0usize;

        for packet_count in 0..=2usize {
            let send_retry_variants = retry_vectors(packet_count, max_retry);
            let recv_retry_variants = retry_vectors(packet_count, max_retry);

            for send_retries in &send_retry_variants {
                for recv_retries in &recv_retry_variants {
                    for initial_present in [false, true] {
                        for initial_delay in 0..=max_retry {
                            if !initial_present && initial_delay != 0 {
                                continue;
                            }

                            for flush_tail_len in 0..=2usize {
                                for flush_tail_delays in retry_vectors(flush_tail_len, max_retry) {
                                    scenario_count += 1;

                                    let mut reads = Vec::with_capacity(packet_count + 1);
                                    for id in 1..=packet_count {
                                        reads.push(ReadOutcome::Packet(id as u32));
                                    }
                                    reads.push(ReadOutcome::Eof);

                                    let mut io = FakeIo::new(reads);
                                    let mut expected_out = Vec::<u8>::new();

                                    if initial_present {
                                        io.push_initial_delayed(pkt(50), initial_delay);
                                        expected_out.push(50);
                                    }

                                    for idx in 0..packet_count {
                                        let packet_id = (idx + 1) as u32;
                                        let output_id = 100 + idx as u8;
                                        io.eagain_before_accept
                                            .insert(packet_id, send_retries[idx]);
                                        io.push_output_delayed(
                                            packet_id,
                                            pkt(output_id),
                                            recv_retries[idx],
                                        );
                                        expected_out.push(output_id);
                                    }

                                    for (idx, delay) in
                                        flush_tail_delays.iter().copied().enumerate()
                                    {
                                        let output_id = 200 + idx as u8;
                                        io.push_flush_tail_delayed(pkt(output_id), delay);
                                        expected_out.push(output_id);
                                    }

                                    let mut machine = BsfMachine::<u32>::default();
                                    let mut observed_out = Vec::<u8>::new();
                                    loop {
                                        match machine.poll(&mut io).expect("poll should succeed") {
                                            Some(pkt) => observed_out.push(pkt.data[0]),
                                            None => break,
                                        }
                                    }

                                    assert_eq!(
                                        observed_out, expected_out,
                                        "output order/loss mismatch: packets={packet_count} send={send_retries:?} recv={recv_retries:?} initial_present={initial_present} initial_delay={initial_delay} flush_tail={flush_tail_delays:?}"
                                    );

                                    let flush_count = io
                                        .send_log
                                        .iter()
                                        .filter(|item| matches!(item, SendInput::Flush))
                                        .count();
                                    assert_eq!(
                                        flush_count, 1,
                                        "flush should be sent exactly once: packets={packet_count} send={send_retries:?} recv={recv_retries:?} flush_tail={flush_tail_delays:?}"
                                    );

                                    let mut expected_send_log = Vec::<SendInput<u32>>::new();
                                    for idx in 0..packet_count {
                                        let packet_id = (idx + 1) as u32;
                                        for _ in 0..=send_retries[idx] {
                                            expected_send_log.push(SendInput::Packet(packet_id));
                                        }
                                    }
                                    expected_send_log.push(SendInput::Flush);
                                    assert_eq!(
                                        io.send_log, expected_send_log,
                                        "pending retry/send ordering mismatch: packets={packet_count} send={send_retries:?} recv={recv_retries:?} flush_tail={flush_tail_delays:?}"
                                    );

                                    for _ in 0..3 {
                                        assert!(
                                            machine
                                                .poll(&mut io)
                                                .expect("terminal EOS poll should not fail")
                                                .is_none(),
                                            "terminal EOS should be sticky: packets={packet_count} send={send_retries:?} recv={recv_retries:?} flush_tail={flush_tail_delays:?}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        assert!(
            scenario_count > 0,
            "permutation test should execute at least one scenario"
        );
    }
}
