# VideoForge v2.0

**High-performance AI Video Upscaler & Editor Engine**

VideoForge v2.0 is a next-generation, GPU-accelerated video enhancement engine designed for professional workflows. It leverages NVDEC/NVENC, TensorRT/ONNX, and Rust for maximum performance and deterministic AI upscaling without the overhead of Python or Chromium.

---

## Features

* **GPU-Accelerated Pipeline**: Real-time decoding, inference, and encoding via NVDEC/NVENC and TensorRT/ONNX.
* **Precision & Device Control**: Support for FP16/FP32, per-device configuration, and multi-GPU scaling.
* **VRAM-Aware Buffer Management**: Bucketed buffer pools with RAII tracking for safe large-resolution workflows.
* **Robust CLI**: Full-featured command-line interface for batch processing, model selection, and advanced tuning.
* **Profiling & Metrics**: Per-stage timing, queue monitoring, and throughput logging for optimization.
* **Error Handling & Cancellation**: Graceful shutdown, panic recovery, and robust error bubbling.
* **Testing & Audit**: Synthetic stress tests, determinism validation, and pipeline invariant checks.
* **Future-proof QoL**: Auto-detect input resolution, optional EngineFingerprint for reproducibility, optional GUI support.

---

## Getting Started

### Requirements

* CUDA-compatible GPU (with NVDEC/NVENC support)
* Rust 1.72+
* TensorRT and/or ONNX runtime installed
* FFmpeg (optional, for preprocessing/validation)

### Build

Clone the repository and build the release binary:

```bash
git clone https://github.com/your-org/videoforge.git
cd videoforge
cargo build --release
```

### Run

```bash
./target/release/videoforge -i input.mp4 -o output.mp4 -m model.onnx --precision fp16 --device 0
```

Available flags:

* `-i` input file or folder
* `-o` output file or folder
* `-m` model path (ONNX or TensorRT)
* `-p` precision (`fp16`/`fp32`)
* `-d` GPU device ID
* `--audit` run Phase 7 audit suite
* `--vram-limit` max VRAM in MB

---

## Architecture

1. **CLI Layer**: Handles user intent, validates hardware, and constructs EnginePlan.
2. **Runtime Engine**: ThreadManager spawns inference and encode threads, flume channels connect stages.
3. **Upscale Pipeline**: Decoding → Preprocessing → Model Inference → Postprocessing → Encoding.
4. **Resource Management**: Bucketed buffer pools with RAII VRAM tracking and eviction policies.
5. **Testing & Audit**: Built-in stress tests, deterministic validation, and stream concurrency checks.

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

MIT License — see `LICENSE` file for details

---

## About

VideoForge v2.0 is designed to be a **fully native, high-performance video enhancement engine**, replacing Python-based prototyping pipelines and older Rust implementations. Its goal is deterministic, efficient, and production-ready video AI workflows.
