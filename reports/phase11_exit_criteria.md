# Phase 11 Exit Criteria

Date: 2026-02-18T22:40:23Z

Scope: PR11-6 docs/release hygiene only (no pipeline behavior changes).

Execution context: commands run from `/home/calvin/src/rave`.

## Command
```bash
cargo fmt --check
```

### Output
```text
`cargo metadata` exited with an error: error: could not find `Cargo.toml` in `/home/calvin/src/rave` or any parent directory

This utility formats all bin and lib files of the current crate using rustfmt.

Usage: cargo fmt [OPTIONS] [-- <rustfmt_options>...]

Arguments:
  [rustfmt_options]...  Options passed to rustfmt

Options:
  -q, --quiet
          No output printed to stdout
  -v, --verbose
          Use verbose output
      --version
          Print rustfmt version and exit
  -p, --package <package>...
          Specify package to format
      --manifest-path <manifest-path>
          Specify path to Cargo.toml
      --message-format <message-format>
          Specify message-format: short|json|human
      --all
          Format all packages, and also their local path-based dependencies
      --check
          Run rustfmt in check mode
  -h, --help
          Print help

[exit_code]=1
```

## Command
```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Output
```text
error: could not find `Cargo.toml` in `/home/calvin/src/rave` or any parent directory

[exit_code]=101
```

## Command
```bash
cargo test --workspace
```

### Output
```text
error: could not find `Cargo.toml` in `/home/calvin/src/rave` or any parent directory

[exit_code]=101
```

## Command
```bash
cargo build --workspace --release
```

### Output
```text
error: could not find `Cargo.toml` in `/home/calvin/src/rave` or any parent directory

[exit_code]=101
```

## Summary
- `cargo fmt --check` -> `1` (failed: no workspace manifest at repo root)
- `cargo clippy --workspace --all-targets -- -D warnings` -> `101` (failed: no workspace manifest at repo root)
- `cargo test --workspace` -> `101` (failed: no workspace manifest at repo root)
- `cargo build --workspace --release` -> `101` (failed: no workspace manifest at repo root)
