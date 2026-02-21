# No Host Copies Audit Checklist

## Invariant

In the steady-state GPU-native path, frame payloads must not be copied to host
memory between:

`decode -> preprocess -> inference -> encode`

Allowed exceptions:
- Container I/O packet bytes can live in host memory (compressed bitstream).
- Logging/metrics are host-side.
- Debug-only readbacks are allowed only behind explicit audit/debug feature flags.

## Decode (`rave-nvcodec`)

- [ ] Decoder outputs GPU-resident surfaces only (NV12 or equivalent).
- [ ] No `cudaMemcpyDtoH` or host staging buffers in steady state.
- [ ] Any debug readback is behind `feature = "debug-host-copies"` (or stricter audit gate).

## Preprocess (`rave-cuda`)

- [ ] NV12->RGB (or required format) conversion is a GPU kernel.
- [ ] Normalization/packing stays on GPU (no CPU pixel loops).
- [ ] Buffer allocations use pooled/pre-allocated device buffers.

## Inference (`rave-tensorrt`)

- [ ] Inputs/outputs use ORT IO Binding with device pointers.
- [ ] Pointer identity is audited in debug/audit builds (input pointer matches `GpuTexture`; output pointer matches ring slot).
- [ ] Provider policy avoids CPU EP fallback, or strict mode fails fast when fallback would break no-host-copies guarantees.

## Encode (`rave-nvcodec`)

- [ ] Encoder consumes GPU-resident surfaces only.
- [ ] No CPU color conversion in steady state.
- [ ] Output compressed packets may be copied to host (acceptable boundary).

## Mux/Demux (`rave-ffmpeg`)

- [ ] Operates on compressed packets only (host memory acceptable).
- [ ] Does not decode raw frames or use rawvideo frame APIs.

## Pipeline (`rave-pipeline`)

- [ ] Bounded channels connect stages; no stage serializes GPU frame payloads to host for transport.
- [ ] `PipelineConfig::strict_no_host_copies` is available for strict audit runs.
- [ ] Audit feature wiring is enabled with `--features audit-no-host-copies` when validating invariants.

## Determinism Contract (`production_strict`)

Definition:
- Same input + same model + same profile + same device should produce the same
  canonical stage output bytes.
- Container bytes may differ (timestamps/metadata), so determinism is audited
  at canonical stage checkpoints rather than container payload hashes.

Checkpoint policy:
- Checkpoint hashing is optional and controlled by `RunContract`.
- Host readback for checkpoint hashing is only allowed behind
  `feature = \"debug-host-copies\"`.
- In strict mode, requested checkpoint hashing without that feature is reported
  as an audit warning/failure according to contract policy.

## How To Run Audits

- Dependency boundaries: `./scripts/check_deps.sh`
- Standard checks: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
- Strict audit build: `cargo test --workspace --features audit-no-host-copies`
- Determinism checkpoints (debug readback enabled): `cargo test --workspace --features \"audit-no-host-copies debug-host-copies\"`
