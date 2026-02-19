# v0.2 Release Hygiene

## Scope
- Workspace version alignment.
- Crate package metadata validation for publishable crates.
- Publish sequencing and feature-flag check.

## Workspace version
- `workspace.package.version = "0.2.0"` in `Cargo.toml`.
- Member crates use `version.workspace = true`.

## Publishable crates and metadata
- `crates/rave-core`
- `crates/rave-cuda`
- `crates/rave-ffmpeg`
- `crates/rave-nvcodec`
- `crates/rave-tensorrt`
- `crates/rave-pipeline`
- `rave-cli`

Each above now has:
- `name`
- `version.workspace = true`
- `license.workspace = true`
- `repository.workspace = true`
- `readme` (explicit path to root `README.md`)

## Publish order
Use `--locked` and publish in dependency order:

1. `cargo publish -p rave-core --locked`
2. `cargo publish -p rave-cuda --locked`
3. `cargo publish -p rave-ffmpeg --locked`
4. `cargo publish -p rave-nvcodec --locked`
5. `cargo publish -p rave-tensorrt --locked`
6. `cargo publish -p rave-pipeline --locked`
7. `cargo publish -p rave-cli --locked`

## Feature flags check
- `rave-core`: `default`, `debug-alloc`
- `rave-cuda`: no custom features
- `rave-ffmpeg`: no custom features
- `rave-nvcodec`: no custom features
- `rave-tensorrt`: `default`, `debug-alloc`
- `rave-pipeline`: `default`, `debug-alloc`
- `rave-cli`: `default`, `debug-alloc` (enables debug-alloc across dependent crates)

## Notes
- `legacy/engine-v2` is excluded from workspace release flow.
- Recommended pre-publish checks:
  - `cargo fmt --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo build --workspace --release`
