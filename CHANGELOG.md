# Changelog

## [0.4.0] - 2026-02-26

### Added
- `keywords` and `categories` to all published crates for crates.io
  discoverability.
- `homepage.workspace = true` and `rust-version.workspace = true` to all
  crates; `rust-version = "1.85"` added to `[workspace.package]`.
- `[workspace.lints]` table (`missing_docs = "warn"`, `clippy::all = "warn"`);
  all crates opt in via `[lints] workspace = true`.
- `[package.metadata.docs.rs]` with `default-target =
  "x86_64-unknown-linux-gnu"` on all six library crates.
- Crates.io, docs.rs, and license badge headers on all crate README files.
- API docs clarifications for stage graph schema and pipeline contracts.

### Changed
- `serde` and `serde_json` moved to `[workspace.dependencies]`; `rave-pipeline`
  and `rave-cli` now use `workspace = true` references.
- ARCHITECTURE.md title, pipeline diagram caption, and opening sentence updated
  from "VideoForge v2.0" to "RAVE".
- ARCHITECTURE.md micro-batching lead sentence corrected from "supports
  optional micro-batching" to "defines `BatchConfig` for planned
  micro-batching (not yet implemented)".
- Bumped workspace version to 0.4.0.

### Removed
- `examples/simple_upscale.rs` — referenced non-existent legacy paths.
- Examples section from root `README.md`.
- Empty `[build-dependencies]` section from `rave-nvcodec/Cargo.toml`.
- Empty `[dev-dependencies]` section from `rave-cli/Cargo.toml`.
- `examples/` entry from workspace layout diagram in root `README.md`.

## [0.2.0] - 2026-02-19

### Added
- Workspace release hygiene baseline for the split-crate layout.
- Root-level release notes and publish-order guidance for `v0.2.0`.

### Changed
- Aligned workspace crate versions to `0.2.0`.
- Completed package metadata pass for publishable crates (`name`/`version`/`license`/`repository`/`readme`).
