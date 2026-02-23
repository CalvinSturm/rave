# Windows Integration Checklist

This checklist defines the expected setup for building and testing `third_party/rave`
inside the parent app layout:

```text
third_party/
  ffmpeg/
  nvcodec/
  rave/
```

## Required Toolchain

- Rust stable (`rustup default stable`)
- Visual Studio 2022 Build Tools (MSVC linker + Windows SDK)
- LLVM with `libclang.dll` in `C:\Program Files\LLVM\bin`
- NVIDIA CUDA Toolkit (tested with CUDA 12.x)

## Runtime/Dependency Paths

- `CUDA_PATH` must be set to your CUDA install root.
- FFmpeg is expected at `third_party/ffmpeg` (sibling of `rave`).
- NVCodec libs are resolved in this order:
  1. `third_party/rave/third_party/nvcodec`
  2. `third_party/nvcodec` (parent app layout)
  3. fallback to CUDA Toolkit libs

Workspace `.cargo/config.toml` provides:
- `FFMPEG_DIR` relative path for `ffmpeg-sys-next`
- `LIBCLANG_PATH` for bindgen
- `PKG_CONFIG=nonexistent` to avoid system pkg-config/vcpkg probing on Windows

## Standard Commands (PowerShell)

Build:

```powershell
.\scripts\build.ps1 --workspace
.\scripts\build.ps1 -- -p rave-cli --bin rave --release
```

Test:

```powershell
.\scripts\test.ps1
.\scripts\test.ps1 --features audit-no-host-copies
```

## Expected Success Signals

- Build prints: `Finished 'release' profile` or `Finished 'dev' profile`
- Tests print: `test result: ok` across workspace crates
- No `ffmpeg-sys-next` failure about missing `pkg-config`
- No link errors for CUDA symbols like `cuInit`/`cuDeviceGetCount`

## Common Failure Signatures

- `STATUS_DLL_NOT_FOUND (0xc0000135)` on test binaries:
  ensure PATH includes `third_party/ffmpeg/bin` and `CUDA_PATH\bin`
  (wrapper scripts handle this).
- `Could not run pkg-config ... command could not be found`:
  verify `.cargo/config.toml` exists and includes `PKG_CONFIG=nonexistent`.
- `CUDA_PATH env var must be set`:
  set `CUDA_PATH` to your installed CUDA root.
- `cargo clean` access denied on Windows:
  stop processes locking `target\` artifacts before re-running clean.
