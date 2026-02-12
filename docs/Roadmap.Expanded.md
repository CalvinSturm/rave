# Roadmap Expanded
**Phase 0 — Research & Prototyping**

* Explore NVDEC/NVENC APIs and TensorRT/ONNX inference

  * Understand available GPU decoding and encoding capabilities
  * Identify API constraints, supported resolutions, and hardware limits
* Verify GPU capability for video decoding, encoding, and model execution

  * Check memory bandwidth, compute capacity, and driver compatibility
  * Run small benchmark tests to estimate performance
* Identify alignment, pitch, and memory constraints for NV12 surfaces

  * Determine proper pitch alignment for NV12 buffers
  * Ensure memory layouts satisfy NVENC/NVDEC hardware requirements
* Prototype small “decode → model → encode” loops

  * Implement minimal end-to-end test pipeline
  * Confirm GPU-to-GPU data flow, basic inference, and encoded output correctness

Outcome: confident understanding of hardware constraints and APIs; baseline pipeline prototype ready for CLI integration

---

**Phase 1 — CLI Entrypoint & Engine Bootstrap**

Phase 1A — CLI & User Intent Parsing

* Parse flags/options: -i input, -o output, -m model, -p precision, -d device, --audit, --vram-limit

  * Support both short and long flag variants
  * Validate input paths and flag combinations before execution
* Convert user flags into canonical PipelineConfig

  * Map flags to structured fields: input_path, output_path, model_path, precision, device_id, batch_size, audit_enabled, vram_limit
  * Ensure defaults are set where necessary
    Outcome: user intent is fully captured and validated

Phase 1B — Engine Capability Validation

* Detect hardware & driver capabilities: GPU name, CUDA version, NVDEC/NVENC support, VRAM total

  * Query system for device properties and driver versions
  * Determine if requested precision (F16/F32) is supported
* Validate pipeline feasibility

  * Confirm input resolution meets NVENC alignment requirements
  * Ensure VRAM budget accommodates pipeline depth and buffer pools
  * Check for optional multi-GPU support
* Generate EnginePlan

  * Define pipeline_depth, stream_layout, buffer_pool_sizes, pitch, encoder_profile, model_precision
    Outcome: deterministic plan ensures safe pipeline execution

Phase 1C — Engine Runtime Construction

* Build core runtime objects from validated EnginePlan:

  * GpuContext for device access and streams
  * PreprocessKernels for format conversion and scaling
  * UpscaleBackend (ONNX/TensorRT) for model execution
  * VideoDecoder / VideoEncoder with proper surface formats
  * PipelineConfig with channel depths, precision, pitch
* Optional Phase 7 Audit hook for verification before running pipeline
* Call UpscalePipeline::run() using EnginePlan
  Outcome: program can run safely with full hardware validation

Phase 1D — The Reactor Core

* Define ThreadManager

  * Central controller for managing worker threads
* Spawn dedicated thread for inference (blocking-safe)

  * Ensure model execution does not block main thread
* Spawn dedicated thread for encoding (context-bound)

  * Maintain proper GPU context for NVENC/NVDEC
* Establish flume channels between threads

  * Enable asynchronous communication and zero-copy buffer transfer

---

**Phase 2 — Decoder & Encoder Integration**

* Plug in real video codecs: NVDEC/NVENC GPU-accelerated decode/encode

  * Correct handling of frame timestamps, keyframes, and formats
* Freeze PixelFormat / SurfaceDescriptor: width, height, pitch, chroma_layout, alignment, memory_kind

  * Create immutable descriptors for pipeline stages
* Ensure decoder produces NV12 consistently

  * Validate output format, stride, and memory alignment
* Ensure encoder consumes NV12 correctly

  * Test encoded output for frame accuracy
* Add error handling for codec failures

  * Gracefully propagate errors upstream
    Outcome: GPU pipeline can process real video files end-to-end

---

**Phase 3 — Backend Model Support**

* Load models (ONNX/TensorRT) with proper precision paths (F16/F32)

  * Support multiple backends and automatic precision selection
* Implement device binding, batch handling, and stream layout

  * Ensure threads and streams are efficiently utilized
* Enforce Model IO Layout Contract (e.g., NCHW FP16 RGB planar)

  * Guarantee consistency between preprocessing and inference
* Optional dynamic model selection via CLI flags
  Outcome: pipeline can run real AI upscaling on GPU safely

---

**Phase 4 — Resource Management & VRAM**

* Respect VRAM limits (--vram-limit) from EnginePlan

  * Prevent oversubscription that could crash the pipeline
* Implement bucketed buffer pools with explicit eviction policy

  * Reuse memory efficiently, reduce allocations
* VRAM usage must be tracked by RAII, not by manual decrements

  * Automatic memory accounting ensures correctness under failures
* Ensure pipeline does not exceed memory caps for large resolutions
* Validate pool hit rate ≥ 90% in stress tests
* Handle stream synchronization gracefully
  Outcome: engine is stable for large videos and heavy inference

---

**Phase 5 — Profiling & Metrics**

* Enable per-stage timing: decode → preprocess → inference → postprocess → encode
* Track queue depths for each stage
* Optional integration of real-time GPU profiler
* Log throughput, VRAM usage, frame counts at shutdown
* Use logs to tune pipeline depth versus latency
  Outcome: performance bottlenecks are visible and tunable

---

**Phase 6 — Error Handling & Cancellation**

* Support Ctrl+C / SIGINT graceful cancellation
* Drop senders, flush encoder, sync streams
* Propagate errors from pipeline stages to main binary
* Panic recovery via EngineError::PanicRecovered
  Outcome: pipeline is robust and non-blocking on failure

---

**Phase 7 — Testing & Audit**

* Run synthetic stress tests (UpscalePipeline::stress_test_synthetic)

  * Validate frames_decoded == frames_encoded
  * No VRAM leaks
  * Pool hit rate ≥ 90%
  * Concurrency is correct
* Run Phase 7 Audit Suite

  * Zero host allocations
  * Determinism (VRAM delta small)
  * Stream overlap/concurrency checks
* Optional unit tests per stage
  Outcome: pipeline is verified, deterministic, and production-safe

---

**Phase 8 — Packaging & Distribution**

* Build release binary (cargo build --release)
* Bundle model and runtime dependencies
* Optional Docker container for reproducibility
* Optional automatic flag validation & help text
  Outcome: users can run videoforge -i input.mp4 -o output.mp4 -m model.onnx out-of-the-box

---

**Phase 9 — Quality-of-Life Enhancements**

* Auto-detect input resolution & pitch
* Multi-GPU support (optional)
* Batch inference tuning (throughput vs latency tradeoff)
* Logging verbosity flags (--debug, --trace)
* Optional GUI for non-CLI users
* Optional EngineFingerprint for reproducibility & bug reports: GPU name, driver version, CUDA version, TensorRT version, model hash
  Outcome: fully usable, configurable, and production-ready engine