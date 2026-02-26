[![Crates.io](https://img.shields.io/crates/v/rave-ffmpeg.svg)](https://crates.io/crates/rave-ffmpeg)
[![docs.rs](https://docs.rs/rave-ffmpeg/badge.svg)](https://docs.rs/rave-ffmpeg)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

# rave-ffmpeg

FFmpeg-based container I/O and probing for RAVE.

`rave-ffmpeg` provides demux/mux/probe utilities and raw file bitstream
adapters used by CLI and pipeline wiring.

## Scope

- Container probing (`probe_container`)
- Container demux to Annex B packets (`FfmpegDemuxer`)
- Container mux from encoded packets (`FfmpegMuxer`)
- Raw file adapters (`FileBitstreamSource`, `FileBitstreamSink`)

## Public API Highlights

- `probe_container(path) -> ContainerMetadata`
- `FfmpegDemuxer::new(path, codec)`
- `FfmpegMuxer::new(path, width, height, fps_num, fps_den)`
- `ContainerMetadata` with codec/dimensions/fps/time-base/duration

## Typical Usage

```rust,no_run
use std::path::Path;

use rave_core::error::Result;
use rave_ffmpeg::probe_container;

fn inspect(path: &Path) -> Result<()> {
    let meta = probe_container(path)?;
    println!("codec={:?} {}x{} fps={}/{}", meta.codec, meta.width, meta.height, meta.fps_num, meta.fps_den);
    Ok(())
}
```

## Runtime Requirements

- FFmpeg shared libraries discoverable by linker/runtime
  (`avcodec`, `avformat`, `avutil`, etc.)

## Notes

- Demux path converts H.264/H.265 MP4-style payloads to Annex B when needed.
- Mux path writes HEVC into container formats inferred from output extension.
- Raw file source/sink are useful for simple bitstream workflows and testing.

## Recent Hardening

- BSF state machine now preserves packets across `EAGAIN` and retries safely.
- EOF handling now performs explicit BSF flush + drain before terminal `None`.
- Added deterministic unit tests for `EAGAIN` retry and flush-drain behavior.
- Added a boundary test that guards against raw-frame decode API usage in demux/mux paths.
