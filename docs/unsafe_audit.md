# Unsafe Boundary Audit Checklist

This document tracks high-risk unsafe/FFI boundaries and the invariants we rely on to keep behavior defined and recoverable.

## Hotspot Inventory

| Area | Crate / file(s) | Unsafe/FFI API used | Invariants relied on (ownership/lifetime/alignment/order) | Failure mode if violated | Existing tests/guards | Next hardening step |
| --- | --- | --- | --- | --- | --- | --- |
| ORT tensor binding + provider loader bridge | `crates/rave-tensorrt/src/tensorrt.rs` | ORT C API tensor construction from device memory, `dlopen` for provider DSOs | Bound pointers must remain valid for the full ORT call; bound IO pointers must match ring/input pointers exactly; provider bridge (`providers_shared`) must be globally visible before TensorRT provider load | Host staging/copy regression, invalid pointer deref, provider load failure or fallback mismatch | Pointer-identity audit and tests, batch guardrails, provider preload smoke tests, no-host-copy violation hook | Add a narrow fuzzable harness for tensor-shape/byte-size mismatch paths |
| NVDEC mapped-surface lifecycle + async D2D + unmap ordering | `crates/rave-nvcodec/src/nvdec.rs` | `cuvidMapVideoFrame64`, `cuMemcpy2DAsync_v2`, `cuEventRecord`, `cuvidUnmapVideoFrame64`, parser callbacks | Map/unmap pairing must be exact; mapped pointer must not outlive unmap; copy completion event must be recorded after D2D and waited on before downstream read; event handles must be recycled exactly once | Use-after-unmap, stale/incomplete frame reads, decode deadlock/stall, resource leaks | Event pool + recycle channel, explicit cross-stream wait helper, queue-depth accounting, decode lifecycle tests in pipeline flow | Add decode stress test with forced cancellation at multiple callback boundaries |
| NVENC register/map/unmap/lock/unlock ordering | `crates/rave-nvcodec/src/nvenc.rs` | NVENC API (`nvEncRegisterResource`, `nvEncMapInputResource`, `nvEncEncodePicture`, `nvEncLockBitstream`, `nvEncUnlockBitstream`, `nvEncUnmapInputResource`) | Resource registration cached per device ptr; every map must be followed by unmap; lock must always be followed by unlock even on sink errors; EOS flush ordering must be respected | Encoder corruption, leaked mapped resources, deadlock in encode thread, truncated bitstream | Single-threaded encoder usage, explicit unlock/unmap on path, drop-time unregister loop | Add targeted failure-injection test seam around sink write errors with map/lock counters |
| FFmpeg AVPacket ownership through EAGAIN/EOF/flush | `crates/rave-ffmpeg/src/ffmpeg_demuxer.rs` | `av_read_frame`, `av_bsf_send_packet`, `av_bsf_receive_packet`, `av_packet_unref`, `av_packet_move_ref` | `pkt_read`/`pkt_pending` handoff must preserve ownership on EAGAIN; flush sent once on source EOF; drain until BSF EOF before terminal EOS; packet data copied before unref | Packet loss/duplication, stuck demux loop, leaked AVPacket refs, corrupted packet stream | Deterministic unit tests for EAGAIN retry and flush-drain; permutation/property tests for interleavings and terminal EOS stability | Add mutation corpus for malformed packet metadata (size/data ptr invariants) |
| CUDA driver init + WSL loader path + stream/event sync | `crates/rave-core/src/context.rs`, `crates/rave-cuda/src/stream.rs`, `rave-cli/src/main.rs` | CUDA driver FFI (`cuInit`, `cuStreamSynchronize`, `cuStreamWaitEvent`), `dlopen`/`dladdr` loader probing and WSL preload | Correct libcuda origin in WSL; idempotent init/retry behavior; stream handles must stay context-local; events only used with valid stream/context pairs; Linux re-exec must avoid loops | Startup failure, WSL symbol conflicts, stream sync UB/deadlock, hidden partial initialization | Loader diagnostics + WSL guards, Linux re-exec gating tests, stream wait helpers, strict/audit pipeline modes | Add targeted unit tests for additional loader edge cases (env precedence + re-exec guard interactions) |

## PR Review Checklist (Unsafe Changes)

- What invariant changed, and where is it documented in this file?
- What new or updated automated test proves the invariant now holds?
- If this change regresses in production, what is the rollback plan (single commit/flag/path)?

## Run Sanitizers Locally

```bash
# Linux prerequisites (Debian/Ubuntu)
sudo apt-get update
sudo apt-get install -y clang llvm pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

# Nightly toolchain for -Zsanitizer
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly

# AddressSanitizer on CPU-only ffmpeg surfaces
RUSTFLAGS="-Zsanitizer=address -Cpanic=abort" \
RUSTDOCFLAGS="-Zsanitizer=address" \
ASAN_OPTIONS="detect_leaks=1,strict_string_checks=1,check_initialization_order=1" \
cargo +nightly test -Zbuild-std --target x86_64-unknown-linux-gnu -p rave-ffmpeg --lib
```
