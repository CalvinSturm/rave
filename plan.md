# Implementation Plan: VideoForge v2.0 Documentation Accuracy Pass

## Audit findings driving this plan

A source-code audit against the current documentation revealed these issues:

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 1 | **Critical** | Micro-batching documented as implemented, but `BatchConfig` is a stub — inference always processes single frames. `process()` in `tensorrt.rs` has no batch collection loop. | README.md:53, ARCHITECTURE.md:61–72 |
| 2 | **Moderate** | "Postprocess + Encode" described as one stage, but postprocess (RGB→NV12 kernel) actually runs inside the inference task, not the encode task. Pipeline has 4 concurrent tasks but 5 logical transforms. | README.md:93 |
| 3 | **Minor** | `documentation` field in Cargo.toml points to `docs.rs/videoforge-engine` — this URL won't resolve until the crate is published. Harmless but misleading for local contributors. | Cargo.toml:9 |
| 4 | **Minor** | Performance snapshot table has no caveat that numbers are extrapolated/estimated, not measured from the codebase's own benchmark suite (which doesn't exist). | README.md:30–42 |

All other documented claims (GpuTexture ownership, CLI flags, QueueDepthTracker updates, postprocess kernel existence, channel capacities) were **verified correct** against source.

---

## Step 1 — README.md: Fix micro-batch claim and postprocess staging

### 1a. Fix micro-batch feature bullet (line 53)

**Current (false):**
```
- **Micro-batch support** — configurable `BatchConfig` with `max_batch` and
  `latency_deadline_us` (default: single-frame, 8 ms deadline) for trading
  latency against throughput
```

**Replacement:**
```
- **Micro-batch configuration (planned)** — `BatchConfig` defines `max_batch`
  and `latency_deadline_us` fields; current implementation processes single
  frames. Multi-frame batch dispatch is a planned optimization.
```

Rationale: The struct exists and is wired through the constructor, so it's fair to mention. But stating it "supports" batching is a verifiable lie — the `process()` function never accumulates frames.

### 1b. Fix "Postprocess + Encode" data-flow description (line 93)

**Current (imprecise):**
```
4. **Postprocess + Encode** — CUDA kernel converts RGB F32 → NV12, NVENC encodes to H.264/HEVC bitstream
```

**Replacement:**
```
4. **Postprocess** — CUDA kernel converts upscaled RGB F32 → NV12 (runs inside the inference task)
5. **Encode** — NVENC encodes NV12 to H.264/HEVC bitstream (separate blocking task)
```

And update the summary line above this list from "four stages" to "four concurrent tasks spanning five logical transforms" or similar phrasing that matches reality.

### 1c. Soften performance snapshot framing (lines 30–42)

Add a caveat making clear these are illustrative estimates, not regression-tested benchmarks:

```
> **Note:** These are illustrative estimates from development testing, not
> CI-reproducible benchmarks. VideoForge does not yet include an automated
> benchmark suite.
```

### 1d. Move "Batch inference" into project-status "What's next" section

Remove the misleading feature bullet and ensure the "What's next" section (line 162) covers batch inference as a planned item. Currently it already says:
```
- Batch inference (process N frames per TensorRT invocation)
```
This is correct and should remain. The fix is purely removing the false "implemented" claim from the features list.

---

## Step 2 — Cargo.toml: Minor metadata hygiene

### 2a. Documentation URL: conditionally useful

The `documentation = "https://docs.rs/videoforge-engine"` field is only useful after crates.io publication. Two options:

- **Option A (recommended):** Keep it. docs.rs will resolve once published. Pre-publication, contributors use `cargo doc --open` locally. No harm.
- **Option B:** Remove until first publish.

Plan: Keep as-is. No change needed.

### 2b. Verify all other metadata fields

| Field | Current value | Status |
|-------|---------------|--------|
| `name` | `videoforge-engine` | Correct |
| `version` | `2.0.0` | Correct |
| `edition` | `2024` | Correct (Rust 1.85+) |
| `description` | Accurate one-liner | Correct |
| `license` | `MIT` | Matches LICENSE file |
| `repository` | GitHub URL | Correct |
| `homepage` | GitHub URL | Correct |
| `documentation` | docs.rs URL | Acceptable (see 2a) |
| `keywords` | 5 keywords | At crates.io maximum |
| `categories` | `multimedia::video`, `computer-vision` | Correct |
| `authors` | Name + email | Correct |
| `readme` | `../README.md` | Correct relative path for workspace |

**No Cargo.toml changes required.** All fields are accurate and at their limits.

---

## Step 3 — lib.rs: Fix micro-batch doc claim

### 3a. Fix backends module doc (lines 36–40)

**Current (implies implemented):**
```
//!   Includes [`backends::tensorrt::BatchConfig`] for optional micro-batch
//!   collection with a latency deadline.
```

**Replacement:**
```
//!   Defines [`backends::tensorrt::BatchConfig`] for planned micro-batch
//!   collection (not yet implemented; current inference is single-frame).
```

### 3b. All other lib.rs content verified correct

- Pipeline diagram: Accurate (4 stages, correct channel capacities)
- Module descriptions: Match actual module contents
- Design principles: All verified against source
- Safety section: Correctly describes all three FFI boundaries
- Module exports and `#[global_allocator]`: Untouched code, no changes needed

---

## Step 4 — ARCHITECTURE.md: Fix micro-batching section

### 4a. Reframe "Micro-batching" section (lines 61–72)

**Current title:** `## Micro-batching`

**Replacement title:** `## Micro-batching (planned)`

**Current body claims** batch collection is implemented. Replace with:

```markdown
## Micro-batching (planned)

The TensorRT backend defines a `BatchConfig` struct for future multi-frame batching:

\`\`\`rust
pub struct BatchConfig {
    pub max_batch: usize,         // default: 1 (single-frame)
    pub latency_deadline_us: u64, // default: 8000 (8 ms)
}
\`\`\`

**Current behavior:** All inference is single-frame (`max_batch = 1`). The
`process()` method on `TensorRtBackend` accepts and infers one `FrameEnvelope`
at a time. The `latency_deadline_us` field is not yet checked.

**Planned behavior:** When `max_batch > 1`, the inference stage will collect
up to `max_batch` frames (or flush early at the latency deadline) before
dispatching a single batched TensorRT invocation.
```

### 4b. All other ARCHITECTURE.md content verified correct

- Pipeline diagram and stage model: Accurate
- Backpressure section: Verified with code (send().await blocks, queue depths updated)
- Module dependency graph: Matches actual imports
- Data flow table: Pixel formats match kernel source
- Memory model: Bucketed pool, VRAM accounting, ownership all verified
- Concurrency model: spawn_blocking/spawn usage matches pipeline.rs
- CUDA streams table: Matches GpuContext fields
- Cross-stream sync: Verified (CUDA events between stages)
- Telemetry table: All metric structs exist with documented counters

---

## Step 5 — Verify compilation

After all edits:

```bash
cd engine-v2
cargo check          # no code changes, only doc changes — should pass
cargo doc --no-deps  # verify rustdoc renders without warnings
```

---

## Summary of changes

| File | Change | Lines affected | Risk |
|------|--------|---------------|------|
| `README.md` | Fix micro-batch bullet to say "planned" | ~53 | None (text only) |
| `README.md` | Split "Postprocess + Encode" into two numbered items | ~93–94 | None (text only) |
| `README.md` | Add benchmark caveat to performance snapshot | ~42 | None (text only) |
| `lib.rs` | Fix backends doc to say BatchConfig is planned | ~39–40 | None (doc comment only) |
| `ARCHITECTURE.md` | Reframe micro-batching section as planned | ~61–72 | None (text only) |
| `Cargo.toml` | No changes | — | — |

**Total risk: Zero.** All changes are documentation-only. No code, no Cargo.toml, no feature flags.

---

## Files NOT modified

- `Cargo.toml` — all fields verified correct, at crates.io limits, no changes needed.
- `LICENSE` — correct MIT text, correct author and year.
- Any `.rs` source files beyond doc comments in `lib.rs`.
- `docs/Technical-Audit.md` — internal reference, stays as-is.
