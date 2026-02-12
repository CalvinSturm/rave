# VideoForge v2.0 — GPU-Native Engine Replacement Plan

## Executive Summary

v1.0 failed due to architectural violations of physical memory topology:

* SHM race conditions (X-2)
* Cross-language schema drift (X-3)
* VRAM lifecycle divergence (G-1)
* CPU ↔ GPU PCIe churn
* Implicit allocator growth
* Unbounded concurrency

v2.0 is a **single-process, single-context, GPU-resident Rust engine**.

No Python.
No SHM.
No host-backed inference tensors.
No CPU frame staging.

Frames must move strictly:

```
NVDEC → Device Preprocess → TensorRT (ORT IO Binding) → NVENC
```

Zero exceptions.

---

# What Has Already Been Completed

## 1. Architectural Decisions (ADR-Level)

* Removed hybrid Rust/Python architecture
* Eliminated SHM
* Eliminated Zenoh
* Eliminated Torch runtime
* Adopted ORT + TensorRTExecutionProvider
* Enforced zero-copy GPU residency

---

## 2. Hard System Constraints Defined

* Single long-lived CUDA context
* No per-frame context creation
* No host frame buffers
* No `Vec<u8>` for frame data
* Bounded `mpsc` channels
* Explicit backpressure
* No unbounded queues
* No CPU decode fallback

---

## 3. API Contract Defined

* `GpuTexture` structure defined
* `UpscaleBackend` trait defined
* Module boundaries defined
* Required modules enumerated

---

## 4. Safety Philosophy Established

* `unsafe` only for CUDA FFI
* Document invariants per unsafe block
* Deterministic drop order
* Explicit resource ownership

This foundation is correct and complete.

---

# PHASE 0 — Memory Topology Definition (Pre-Implementation Validation)

## Objective

Formalize the GPU memory lifecycle before writing code.

## Tasks

* Define memory graph:

  * NVDEC output surfaces
  * Intermediate preprocess buffers
  * Model input buffers
  * Model output buffers
  * NVENC input surfaces

* Define ownership model:

  * All device allocations originate from a single CUDA context
  * All buffers reference-counted via `Arc`
  * No hidden allocations inside inference calls

* Define stream topology:

  * One decode stream
  * One preprocess stream
  * One inference stream
  * One encode stream
  * OR single unified stream (decision gate)

* Define maximum in-flight frame count:

  * Pipeline depth N (e.g., 2–4 max)
  * Bounded by channel capacity

## Deliverables

* Memory topology diagram
* Stream ownership document
* Concurrency upper bound defined

## Definition of Done

No ambiguous ownership remains.

---

# PHASE 1 — Core GPU Primitives

## Objective

Build deterministic GPU abstractions without inference.

## Modules

### `src/core/types.rs`

* `GpuTexture`
* `PixelFormat`
* RAII invariants
* Context safety documentation
* `Send + Sync` validation

### `src/core/context.rs`

* Create single CUDA context
* Store in `Arc`
* Explicit stream creation
* Context lifetime > all textures

## Hard Requirements

* No implicit context creation
* No thread-local contexts
* Explicit `Drop` order

## Risk

Accidental context duplication via async runtime.

## Mitigation

Context owned by root `Engine` struct and passed by `Arc`.

## Definition of Done

* Can allocate GPU buffer
* Can clone `GpuTexture`
* Deterministic drop order
* No host allocations

---

# PHASE 2 — NVDEC + NVENC GPU Integration

## Objective

Establish GPU-only decode/encode path before inference.

## Tasks

* Integrate NVDEC:

  * Decode directly into CUDA device memory
  * Output NV12 `GpuTexture`

* Integrate NVENC:

  * Accept NV12 `GpuTexture`
  * No host staging

* Validate:

  * Decode → Encode loop
  * Zero host copies confirmed

## Observability

* GPU memory accounting
* PCIe transfer profiling

## Definition of Done

Working GPU-only transcode pipeline.

---

# PHASE 3 — CUDA Preprocessing Kernels

## Objective

Implement GPU-only transforms.

### Required Transforms

* NV12 → RGB
* RGB → FP16 normalize
* Layout conversion NHWC → NCHW
* Batch dimension injection

## Requirements

* No thrust
* No host-side transforms
* Kernel launch reuse
* No per-frame PTX recompilation

## Risks

* Hidden sync points
* Implicit device-host synchronization

## Mitigation

* Explicit stream usage
* No `cudaDeviceSynchronize` except controlled points

## Definition of Done

Input NV12 texture → model-ready tensor buffer (device only).

---

# PHASE 4 — ORT + TensorRT IO Binding Integration

## Objective

Implement GPU-resident inference.

## Required Guarantees

* Use `TensorRTExecutionProvider`
* Use ORT IO Binding
* Bind input/output via device pointers
* No default allocator usage
* No host tensor fallback

## Tasks

* Load model

* Query model metadata:

  * Input shape
  * Dynamic output shape

* Preallocate output buffer pool

* Bind input pointer

* Bind output pointer

* Execute inference

* Return GPU texture

## Risk Areas

* ORT silently allocating host buffers
* Fallback to CPU EP
* Shape mismatch causing hidden reallocation

## Mitigation

* Disable CPU EP explicitly
* Assert device binding
* Validate memory type

## Definition of Done

Inference runs with:

* Zero host allocations
* Stable VRAM usage under sustained load
* No per-frame allocator growth

---

# PHASE 5 — Pipeline Orchestration

## Objective

Wire bounded async pipeline.

```
Decoder → Preprocess → Backend → Encoder
```

## Requirements

* `tokio::sync::mpsc` (bounded)
* Backpressure propagation
* No frame dropping
* No sleep loops
* No busy wait
* Explicit cancellation

## Concurrency Model

* Fixed channel capacity (2–4)
* Each stage awaits send
* Shutdown token propagated

## Risks

* Deadlock via incorrect drop ordering
* Memory growth if sender not awaited

## Mitigation

* Drop senders first
* Join all tasks
* Explicit shutdown barrier

## Definition of Done

* Sustained 4K60 load
* No memory growth
* No deadlock
* Clean shutdown

---

# PHASE 6 — VRAM Lifecycle Stabilization

## Objective

Eliminate allocator churn.

## Tasks

* Introduce GPU buffer pool
* Reuse inference buffers
* Reuse preprocess staging buffers
* Fixed upper memory bound

## Validation

* 30-minute stress test
* VRAM stable within fixed envelope
* No fragmentation growth

---

# PHASE 7 — Determinism & Safety Audit

## Objective

Prove invariants.

## Checks

* No `Vec<u8>` for frames
* No host-backed ORT tensors
* No unbounded channels
* No `cudaDeviceReset`
* No context recreation
* No per-frame stream allocation

## Tooling

* `grep` audit
* Runtime allocation logging
* CUDA memtrace
* Valgrind for host alloc detection

## Definition of Done

Passes internal audit checklist.

---

# PHASE 8 — Performance Tuning

## Focus Areas

* Stream overlap
* Kernel fusion
* Memory alignment
* TensorRT FP16 / INT8 calibration
* Batch pipelining (if viable)

## Guardrail

Performance improvements must not violate memory determinism.

---

# PHASE 9 — Production Hardening

## Tasks

* Error propagation via `anyhow`
* Library errors via `thiserror`
* Deterministic panic boundaries
* Logging with minimal allocation

## Structured Metrics

* Frame latency
* VRAM usage
* Queue depth
* Inference time

---

# Architectural Invariants (Must Hold Forever)

* Single CUDA context
* Frames never enter CPU memory
* No unbounded memory growth
* Backpressure always propagates
* All GPU memory explicitly owned
* Deterministic drop order
* No implicit device synchronization
* No hidden allocators

---

# Concurrency Envelope

Max in-flight frames:

```
channel_capacity + stage_overlap
```

Memory upper bound:

```
VRAM = (input_buffers + preprocess_buffers + model_io_buffers) * pipeline_depth
```

Must be statically reasoned.

---

# Failure Classes Eliminated vs v1.0

| v1.0 Failure            | v2.0 Elimination Mechanism |
| ----------------------- | -------------------------- |
| SHM corruption          | No SHM                     |
| Schema drift            | Single language            |
| Torch RCE risk          | No Torch                   |
| CPU ↔ GPU churn         | Zero copy                  |
| VRAM divergence         | Centralized allocator      |
| Unbounded memory growth | Bounded channels           |
| Context duplication     | Single context owner       |

---

# Execution Order (Realistic Build Sequence)

1. Context + `GpuTexture` primitives
2. NVDEC/NVENC GPU-only pass
3. CUDA preprocess kernels
4. ORT IO binding
5. Pipeline wiring
6. Buffer pooling
7. Stress testing
8. Performance tuning
9. Audit validation

---

# Critical Success Metrics

* 4K60 stable for 30 minutes
* <5% VRAM variance under load
* Zero host copies confirmed via profiling
* Deterministic shutdown
* No memory growth
* No deadlocks

---

# Final Strategic Note

This is not a refactor.
This is a topology correction.

v1.0 violated physical memory constraints.

v2.0 must reflect hardware reality:

* PCIe is expensive
* VRAM is finite
* Contexts are heavy
* Allocators fragment
* Async can deadlock

If respected, this engine will be:

* Faster
* Safer
* Deterministic
* Auditable
* Competitive with Topaz at the systems level