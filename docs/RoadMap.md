# VideoForge Engine Roadmap — Phase 0 → 9 (Improved)

---

Phase 0 — Research & Prototyping

* Explore NVDEC/NVENC APIs and TensorRT/ONNX inference
* Verify GPU capability for video decoding, encoding, and model execution
* Identify alignment, pitch, and memory constraints for NV12 surfaces
* Prototype small “decode → model → encode” loops

Outcome: confident understanding of hardware constraints and APIs

---

Phase 1 — CLI Entrypoint & Engine Bootstrap

Phase 1A — CLI & User Intent Parsing

* Parse flags/options: -i input, -o output, -m model, -p precision, -d device, --audit, --vram-limit
* Convert user flags into canonical PipelineConfig:

PipelineConfig {
input_path,
output_path,
model_path,
precision,
device_id,
batch_size,
audit_enabled,
vram_limit
}

---

Phase 1B — Engine Capability Validation

* Detect hardware & driver capabilities: GPU name, CUDA version, NVDEC/NVENC support, VRAM total
* Validate pipeline feasibility:

  * Model precision supported on device
  * Input resolution compatible with NVENC alignment
  * VRAM budget sufficient for pipeline depth
  * Multi-GPU availability if requested
* Preflight returns EnginePlan:

EnginePlan {
pipeline_depth,
stream_layout,
buffer_pool_sizes,
pitch,
encoder_profile,
model_precision
}

---

Phase 1C — Engine Runtime Construction

* Build core runtime objects from validated plan:
  GpuContext, PreprocessKernels, UpscaleBackend (ONNX/TensorRT), VideoDecoder, VideoEncoder, PipelineConfig (channel depths, precision, pitch)
* Optional Phase 7 Audit hook before pipeline run
* Call UpscalePipeline::run() using EnginePlan

Outcome: program can run safely with full hardware validation

---

Phase 1D — The Reactor Core

* Define ThreadManager
* Spawn dedicated thread for inference (blocking-safe)
* Spawn dedicated thread for encoding (context-bound)
* Establish flume channels between threads

---

Phase 2 — Decoder & Encoder Integration

* Plug in real video codecs: NVDEC/NVENC GPU-accelerated decode/encode, correct handling of frame timestamps, keyframes, and formats
* Freeze PixelFormat / SurfaceDescriptor: width, height, pitch, chroma_layout, alignment, memory_kind
* Ensure decoder produces NV12 consistently
* Ensure encoder consumes NV12 correctly
* Add error handling for codec failures

Outcome: GPU pipeline can process real video files

---

Phase 3 — Backend Model Support

* Load models (ONNX/TensorRT) with proper precision paths (F16/F32)
* Implement device binding, batch handling, and stream layout
* Enforce Model IO Layout Contract (e.g., NCHW FP16 RGB planar)
* Optional: dynamic model selection via CLI flags

Outcome: pipeline can run real AI upscaling on GPU safely

---

Phase 4 — Resource Management & VRAM

* Respect VRAM limits (--vram-limit) from EnginePlan
* Implement bucketed buffer pools with explicit eviction policy
* VRAM usage must be tracked by RAII, not by manual decrements
* Ensure pipeline does not exceed memory caps for large resolutions
* Validate pool hit rate ≥ 90% in stress tests
* Handle stream synchronization gracefully

Outcome: engine is stable for large videos and heavy inference

---

Phase 5 — Profiling & Metrics

* Enable per-stage timing: decode → preprocess → inference → postprocess → encode
* Track queue depths for each stage
* Optional: integrate real-time GPU profiler
* Log throughput, VRAM usage, frame counts at shutdown
* Helps tune pipeline depth vs latency

Outcome: performance bottlenecks are visible and tunable

---

Phase 6 — Error Handling & Cancellation

* Support Ctrl+C / SIGINT graceful cancellation
* Drop senders, flush encoder, sync streams
* Proper error bubbling from stages to main binary
* Panic recovery via EngineError::PanicRecovered

Outcome: pipeline is robust and non-blocking on failure

---

Phase 7 — Testing & Audit

* Run synthetic stress tests (UpscalePipeline::stress_test_synthetic) to validate:

  * frames_decoded == frames_encoded
  * No VRAM leaks
  * Pool hit rate ≥ 90%
  * Concurrency (no stalls)
* Run Phase 7 Audit Suite to validate invariants:

  * Zero host allocations
  * Determinism (VRAM delta small)
  * Stream overlap / concurrency
* Optional: unit tests for each stage

Outcome: pipeline is verified, deterministic, and production-safe

---

Phase 8 — Packaging & Distribution

* Build release binary (cargo build --release)
* Bundle model and runtime dependencies
* Optional: Docker container for reproducibility
* Optional: automatic flag validation & help text

Outcome: users can run videoforge -i input.mp4 -o output.mp4 -m model.onnx out-of-the-box

---

Phase 9 — Quality-of-Life Enhancements

* Auto-detect input resolution & pitch
* Multi-GPU support (optional, scaling for large videos)
* Batch inference tuning (throughput vs latency tradeoff)
* Logging verbosity flags (--debug, --trace)
* Optional GUI for non-CLI users
* Optional EngineFingerprint for reproducibility & bug reports: GPU name, driver version, CUDA version, TensorRT version, model hash

Outcome: fully usable, configurable, and production-ready engine

---

This roadmap now captures:

* Explicit hardware + model validation
* Deterministic EnginePlan construction
* Surface descriptors and memory contracts
* VRAM-aware allocator policies using RAII
* Metrics, profiling, and testing baked in
* Future-proof QoL features
