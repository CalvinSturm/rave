# RAVE — Zero-Copy GPU Video Inference Pipeline in Rust

**RAVE (Rust Accelerated Video Engine)** is a high-performance, hardware-accelerated video inference engine built in Rust that leverages **:contentReference[oaicite:0]{index=0} NVDEC**, **:contentReference[oaicite:1]{index=1}**, **:contentReference[oaicite:2]{index=2}**, and **:contentReference[oaicite:3]{index=3}** for fully zero-copy video processing pipelines.

Unlike Python-based solutions that rely on subprocesses (e.g. piping through **:contentReference[oaicite:4]{index=4}**) or heavy memory copying between CPU and GPU, RAVE keeps frame data strictly on the GPU. Frames are decoded, preprocessed, inferred, and re-encoded without ever touching system RAM, maximizing throughput for tasks such as super-resolution, interpolation, and restoration.

```text
┌──────────────┐      ┌──────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Input File  │  ➔   │   NVDEC Engine   │  ➔   │  CUDA Preproc  │  ➔   │   TensorRT     │
│  (H.264/HEVC)│      │  (NV12 Surface)  │      │ (NV12 → RGB32) │      │  (ONNX Model)  │
└──────────────┘      └──────────────────┘      └────────────────┘      └──────┬─────────┘
                                                                                │
┌──────────────┐      ┌──────────────────┐      ┌────────────────┐              │
│ Output File  │  🠔   │   NVENC Engine   │  🠔   │  CUDA Postproc │  🠔           │
│ (MP4/MKV)    │      │ (H.264 / HEVC)   │      │ (RGB32 → NV12) │              │
└──────────────┘      └──────────────────┘      └────────────────┘              ▼
````

---

## Why this exists

Most video AI workflows glue FFmpeg to PyTorch via pipes, incurring massive PCIe bus overhead and CPU↔GPU context switching costs. RAVE demonstrates how to build a **production-grade, native Rust video engine** where the CPU only orchestrates control flow while the GPU owns the entire data lifecycle.

RAVE is intended both as:

* A practical engine for high-throughput video ML workloads
* A reference architecture for engineers building low-latency, GPU-resident media systems in Rust

---

## Key Features

* **Hardware Decoding:** Direct NVDEC integration for decoding compressed H.264/HEVC streams directly into GPU memory.
* **TensorRT Inference:** Optimized FP16 / INT8 inference via the ONNX Runtime TensorRT execution provider.
* **Zero-Copy Preprocessing:** Custom CUDA kernels perform NV12 ↔ RGB planar conversion, normalization, and layout transforms entirely in VRAM.
* **Hardware Encoding:** Pipelined NVENC output for efficient video compression.
* **Pure Rust Architecture:** ~8K LOC using bounded async channels, `tokio` executors, and RAII-based FFI safety.

---

## Architecture & Data Flow

RAVE uses a bounded, backpressured pipeline with a bucketed GPU memory pool to control VRAM usage.

1. **Demux:** Container packets are extracted.
2. **Decode:** NVDEC writes frames into semi-planar NV12 CUDA surfaces.
3. **Preprocess:** CUDA kernels convert NV12 → RGB planar `f32` (NCHW).
4. **Inference:** TensorRT executes the neural network on the GPU.
5. **Postprocess:** CUDA kernels convert RGB back to NV12.
6. **Encode:** NVENC compresses the resulting frames into the output bitstream.

A full breakdown of concurrency, memory ownership, and backpressure can be found in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Quick Start

### Prerequisites

* NVIDIA GPU (Pascal or newer)
* CUDA Toolkit 12.x
* NVIDIA Video Codec SDK
* FFmpeg development libraries
* Rust (stable toolchain)

### Installation

```bash
git clone https://github.com/CalvinSturm/rave.git
cd rave
# Ensure CUDA_PATH and ONNX Runtime libraries are available in LD_LIBRARY_PATH
cargo build --release --package rave-engine
```

---

## Usage

```bash
./target/release/rave \
  --input "input.mp4" \
  --output "upscaled.mp4" \
  --model "realesrgan.onnx" \
  --scale 4
```

---

## Project Status

* ✅ **Core Pipeline:** NVDEC → CUDA → TensorRT → NVENC fully operational.
* ✅ **Memory Management:** Bucketed pool allocator with VRAM accounting.
* 🚧 **Audio:** Audio passthrough is experimental.
* 🚧 **UI:** CLI is stable; GUI layer is early-stage.

---

## License

MIT © Calvin Sturm