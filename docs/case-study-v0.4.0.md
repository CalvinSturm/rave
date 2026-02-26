# RAVE v0.4.0 Case Study

## Summary

`v0.4.0` was a boundary-reset release for the `rave` Rust workspace.

The goal was to publish reusable crates to crates.io with clear responsibilities, while removing project-specific functionality from the public API surface. In practice, this meant separating core infrastructure from concrete runtime wiring and hardening the workspace for docs.rs + CI portability.

## Context

`rave` is a GPU/video processing workspace built around:

- shared engine contracts and GPU types
- CUDA kernels and stream helpers
- TensorRT inference integration
- FFmpeg / NVDEC / NVENC codec integration
- async pipeline orchestration

Before `v0.4.0`, some project-specific functionality had leaked into public crates, and concrete runtime composition was too tightly coupled to the orchestration crate.

## Problem

The workspace had three release-quality problems:

1. Public API boundaries were too broad
- app-specific functionality was present in published crates
- `rave-pipeline` mixed orchestration concerns with concrete backend/runtime wiring

2. Native crates were harder to document and validate
- docs.rs and CI environments often lack FFmpeg/TensorRT/NVCodec toolchains
- native build assumptions caused avoidable failures

3. Release workflow friction
- cross-crate publish ordering and dependency cycles needed cleanup
- CI failures (OpenSSL, pkg-config, local env leakage) obscured real regressions

## What I Changed

### 1) Re-defined crate boundaries

- Removed project-specific functionality from published core crates
- Kept `rave-pipeline` focused on orchestration, graph contracts, and execution interfaces
- Added `rave-runtime-nvidia` as the concrete NVIDIA runtime composition crate

`rave-runtime-nvidia` now owns:

- input probing / resolution
- CUDA context + kernel setup
- TensorRT backend initialization
- NVDEC decoder creation
- NVENC encoder creation

This split allows downstream applications to depend on generic orchestration (`rave-pipeline`) and opt into a concrete backend stack (`rave-runtime-nvidia`) explicitly.

### 2) Reduced default coupling and hardened publishability

- Feature-gated backend-heavy paths in `rave-pipeline`
- Tightened backend-specific error coupling in `rave-core` behind features
- Reduced unnecessary public API surface in `rave-cuda`
- Added per-crate package `include` lists for cleaner crates.io tarballs
- Marked `rave-cli` as non-publish (`publish = false`)

### 3) Added docs.rs-safe native crate strategy

Implemented no-default/stub build paths for native crates so documentation and CI checks can run without full native toolchains:

- `rave-ffmpeg`
- `rave-tensorrt`
- `rave-nvcodec`
- `rave-runtime-nvidia`
- `rave-pipeline`

This made docs.rs compatibility intentional instead of accidental.

### 4) Hardened CI and release workflow

Added and fixed CI checks for:

- package content inspection (`cargo package --list`)
- docs.rs-style compatibility checks
- publish dry-run validation (tag-triggered)

Resolved CI portability failures caused by:

- missing OpenSSL dev packages / `pkg-config`
- Windows-local `.cargo/config.toml` environment overrides leaking into Linux CI
- clippy/rustc warning policy mismatches (`missing_docs`, MSRV lint noise)

## Release Engineering Notes

`v0.4.0` also involved release management cleanup:

- yanked an incorrect earlier version
- published crates in dependency order
- removed a publish-time dependency cycle between `rave-pipeline` and `rave-runtime-nvidia`
- retagged the GitHub `v0.4.0` release after post-release CI fixes

This was a practical example of “shipping the architecture” and “shipping the release process” together.

## Outcome

- Published a clean `v0.4.0` crate set to crates.io
- Established clearer long-term crate ownership boundaries
- Improved docs.rs and CI reliability for native-heavy crates
- Reduced risk of app-specific features leaking into reusable libraries

## What This Demonstrates

- Rust crate API design and semver boundary discipline
- Monorepo dependency architecture and refactoring
- Native toolchain portability work (FFmpeg / OpenSSL / pkg-config / CI)
- Release engineering under real constraints (publish order, yanks, tags, CI)

## Next Steps (Post-0.4.0)

- Continue tightening `rave-pipeline` backend-agnostic boundaries
- Add self-hosted Windows GPU CI for trusted branch/manual validation
- Keep `0.4.x` changes focused on stability and polish (avoid unnecessary API churn)
