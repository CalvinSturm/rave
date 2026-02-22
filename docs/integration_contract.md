# RAVE Consumer Integration Contract

This document is the handoff contract for integrating `rave-*` crates into other projects and agent workflows.

## Scope

The contract applies to:
- `rave-core`
- `rave-cuda`
- `rave-tensorrt`
- `rave-nvcodec`
- `rave-ffmpeg`
- `rave-pipeline`
- `rave-cli`

## Architecture Contract

Dependency boundaries are intentional and enforced:

```
rave-core      -> (none)
rave-cuda      -> rave-core
rave-tensorrt  -> rave-core
rave-nvcodec   -> rave-core
rave-ffmpeg    -> rave-core
rave-pipeline  -> rave-core, rave-cuda, rave-tensorrt, rave-nvcodec, rave-ffmpeg
rave-cli       -> rave-core, rave-pipeline
```

Do not add cross-crate edges outside this graph without updating boundary policy and tests.

## Profile And Strictness Contract

`production_strict` means:
- strict invariants enabled
- strict VRAM limit enabled
- strict no-host-copies enabled
- determinism policy is `require_hash`
- ORT re-exec gate enabled for runtime commands that need it

`dev` means best-effort defaults and non-strict behavior.

Strict mapping is centralized in `rave-cli`; downstream integrations must not reimplement profile policy ad hoc.

## Feature Contract

Default build:
- no extra features required

Required for strict no-host-copies guarantees:
- build `rave-cli` with `--features audit-no-host-copies`

Example:

```bash
cargo run -p rave-cli --features audit-no-host-copies --bin rave -- validate --json --profile production_strict --fixture tests/fixtures/validate_production_strict.json
```

If `production_strict` is requested without audit capability, failing fast is expected behavior.

## Runtime Mode Contract

### Non-GPU CI / deterministic smoke path

Use mock mode:

```bash
RAVE_MOCK_RUN=1 cargo run -p rave-cli --features audit-no-host-copies --bin rave -- validate --json --best-effort --profile production_strict --fixture tests/fixtures/validate_production_strict.json
```

This path must:
- avoid CUDA/driver initialization
- preserve config validation and strict-policy mapping
- preserve JSON stdout contract

### GPU runtime path

Requires host environment with NVIDIA driver libraries and runtime dependencies.

On Linux/WSL, missing CUDA/TensorRT provider libraries should produce actionable diagnostics rather than silent fallback in strict contexts.

## CLI Output Contract

In `--json` mode:
- stdout emits exactly one final JSON object
- progress/logging goes to stderr only

Schema stability:
- existing field names/types are stable within `schema_version`
- additive fields are allowed
- no renames/removals without schema version bump

Policy visibility:
- final JSON for `upscale`, `benchmark`, and `validate` includes a top-level `policy` object
- `validate` also reports host-copy audit status fields

## Determinism Contract

Validation determinism policy is explicit:
- `best_effort`: hash may be skipped with reason
- `require_hash`: hash skip is an error

For skipped hashes, reason codes must be stable (machine-readable).

## Micro-Batching Contract

Micro-batching is intentionally not implemented.

`max_batch > 1` must fail fast with actionable messaging. Integrations must treat this as unsupported configuration, not a degraded mode.

## Linux Loader Contract

CUDA linkage is runtime-resolved in CLI/core/cuda paths to avoid hard link failures on non-GPU builders.

Provider/driver resolution errors must include:
- what was being resolved
- candidate paths/strategy
- actionable hint text

## Required Validation For Downstream Integrators

Run these commands in your integration branch:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test
./scripts/check_deps.sh
./scripts/check_docs.sh
```

If your environment has no GPU:
- use `RAVE_MOCK_RUN=1` for strict validate fixture tests
- do not require live CUDA init in generic CI jobs

## Recommended Adoption Checklist

1. Pin to a known-good commit range and record SHA in your project.
2. Decide CI split: non-GPU mock coverage vs GPU runner coverage.
3. Enable `audit-no-host-copies` wherever `production_strict` validation is expected to pass.
4. Parse CLI JSON by schema fields only, never by stderr text.
5. Treat strict failures as contract violations, not flaky retries.

