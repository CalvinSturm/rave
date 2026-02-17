# rave — Phase 10 Execution Contract (Hardened Revision)

## Role & Authority

* You have **commit authority** on the RAVE codebase.
* Your mandate is **Phase 10 only**: library-first crate split, examples, and initial documentation.
* **You may not** alter verified pipeline architecture, zero-copy invariants, public API semantics, or memory behavior.
* **This is an execution task, not a design exercise.**

If ambiguity arises, preserve existing behavior exactly.

---

# System Context (Non-Negotiable Facts)

* RAVE is a **Rust-native, GPU-resident AI video engine**.
* Pipeline stages (bounded):

  1. Decode
  2. Preprocess
  3. Inference
  4. Encode
* Pipeline is **GPU-resident end-to-end** in steady state.
* VRAM managed by **BucketedPool** allocator, **zero-free steady state**.
* Backpressure enforced via **bounded tokio::mpsc channels**.
* CUDA, TensorRT, NVDEC/NVENC, and FFmpeg currently implemented in a **monolithic crate**.

### Verified Behavior (Must Remain Intact)

* Zero-copy GPU residency enforced.
* Pointer identity preserved across stages.
* No per-frame heap allocation in hot paths.
* Deterministic output for identical inputs.

---

# Phase Scope (Phase 10 Only)

## In Scope

* Convert the monolithic crate into a **Cargo workspace**.
* Split into crates **without redesign**:

```
rave/
├── crates/
│   ├── rave-core/
│   ├── rave-cuda/
│   ├── rave-tensorrt/
│   ├── rave-nvcodec/
│   ├── rave-ffmpeg/
│   └── rave-pipeline/
├── rave-cli/
└── Cargo.toml
```

* Add developer-facing examples (compile out-of-the-box):

1. `simple_upscale.rs`
2. `progress_callback.rs`
3. `custom_kernel.rs`
4. `batch_directory.rs`
5. `benchmark.rs`

* Add `rave::prelude` re-exporting:

  * Core GPU types
  * Pipeline builder
  * Common error/result types

---

## Clarified 10-Line Constraint (Hard Requirement)

The “Hello World” upscale must:

* Compile using only `use rave::prelude::*;`
* Contain **≤ 10 non-import, non-comment lines inside `main()`**
* Require no manual backend wiring
* Require no feature flags
* Use only stable public APIs
* Not rely on internal modules

The goal is ergonomic proof — not code golf.

---

# Explicit Non-Goals (Hard Stop)

* No backend abstraction redesign.
* No Metal, Vulkan, CPU, or multi-GPU support.
* No model registry.
* No CLI feature expansion beyond wiring existing behavior.
* No Python, Node.js, WASM, or C bindings.
* No new features or optimizations unless required strictly to preserve parity.

---

# Explicit API Preservation Requirement

## No API Redesign (Hard Constraint)

* Public types must retain their names.
* Public function signatures must remain semantically identical.
* Trait definitions must not be refactored, split, merged, or generalized.
* Generic bounds must not be expanded or abstracted.
* Module paths may change only due to crate relocation.
* Re-exports may be added, but not removed.

Relocation is permitted. Redesign is not.

---

# No New Abstraction Layers (Hard Constraint)

The following are forbidden:

* Introducing new backend traits.
* Introducing generic device abstractions.
* Adding “future-proofing” layers.
* Adding indirection between pipeline stages.
* Introducing polymorphic dispatch where none existed.
* Adding dynamic backend selection logic.

This phase is structural only.

---

# Architectural Constraints (Hard Requirements)

## 1. Zero-Copy Invariant

* No device↔host transfers in steady state.
* No `cudaMemcpy*` in hot paths.
* GPU pointer identity must remain stable:
  Decode → Preprocess → Inference → Encode.

## 2. Allocation Discipline

* No per-frame `Vec`, `Box`, or heap allocation in hot paths.
* All GPU memory allocated via BucketedPool.
* Pool hit rate must remain ≥ 95% under steady load.

## 3. Crate Boundary Enforcement

Dependency graph must remain acyclic:

```
rave-core
↑
rave-{cuda, tensorrt, nvcodec, ffmpeg}
↑
rave-pipeline
↑
rave-cli
```

* `rave-core` must not depend on CUDA, TensorRT, NVDEC/NVENC, FFmpeg, or any FFI.
* Backend crates must not depend on each other.
* Backend types must not leak into `rave-core`.

## 4. Pipeline Correctness

* Bounded backpressure remains intact.
* Queue depths remain observable and bounded.
* Cancellation, shutdown, and error propagation remain identical to monolithic reference.

---

# Invariant Precedence Clause (Critical)

If any conflict arises between:

* Crate boundary restructuring
* API relocation
* Documentation improvements
* Example ergonomics

and the following invariants:

* Zero-copy residency
* Pointer identity
* Allocation discipline
* Determinism
* Backpressure guarantees

**The invariants take absolute precedence.**

If preservation cannot be guaranteed:

* Halt execution.
* Report violation.
* Do not proceed with workaround redesign.

Structural purity is subordinate to runtime integrity.

---

# Verification & Measurement

## Baseline Capture (Before Split)

* FPS @ 1080p
* Peak VRAM usage
* BucketedPool hit rate
* Queue depth per stage
* Per-stage latency
* Reference output SHA256

## Post-Split Validation

* FPS regression ≤ 1%
* VRAM delta ≤ 1%
* Pool hit rate ≥ 95%
* Identical SHA256 hashes
* No new allocations in hot paths
* Zero-copy invariant re-verified

---

# Definition of Done

Phase 10 complete only when:

* `cargo build --workspace` succeeds
* All tests pass
* All examples compile and run
* Rustdoc builds without warnings
* Benchmarks match baseline within tolerance
* Zero-copy, pointer identity, and allocation invariants verified
* Public API surface unchanged except for crate path relocation

---

# Output Requirements

* Show new file tree
* Show crate boundaries
* Show dependency direction
* Show invariant verification notes per major step
* Do not speculate about future phases
* Do not introduce roadmap commentary

---

# Final Instruction

Execute **Phase 10 only**.

Preserve architectural integrity.

Favor determinism over convenience.

No redesign. No abstraction. No scope expansion.

All decisions must be measurable, auditable, and mechanically defensible.
