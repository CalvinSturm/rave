[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

# rave-cli

Command-line interface for the RAVE GPU video pipeline.

`rave-cli` wires all workspace crates into an operator-facing executable with
commands for pipeline execution, benchmarking, and environment probing.

## Commands

- `upscale`: full decode -> preprocess -> infer -> encode path
- `benchmark`: throughput-focused run with optional encode skip/fallback
- `probe`: CUDA/provider probe for runtime diagnostics
- `devices`: list visible CUDA devices

## JSON Contract

- `--json` emits one final JSON object on stdout
- progress is emitted on stderr via `--progress` (`human` or `jsonl`)
- payloads include `schema_version` for parser stability

## Quickstart

```bash
# Human-readable probe
target/debug/rave probe

# JSON probe
target/debug/rave probe --json

# Benchmark without encode
target/debug/rave benchmark --input in.mp4 --model model.onnx --skip-encode

# Benchmark JSON + output file
target/debug/rave benchmark --input in.mp4 --model model.onnx --skip-encode --json --json-out /tmp/bench.json

# Full upscale
target/debug/rave upscale --input in.mp4 --output out.mp4 --model model.onnx
```

## Feature Flags

- `debug-alloc`: enables allocation instrumentation across `rave-core`,
  `rave-pipeline`, and `rave-tensorrt`

## Runtime Requirements

- NVIDIA CUDA driver + toolkit runtime
- NVDEC/NVENC runtime libraries
- FFmpeg runtime libraries
- ONNX Runtime + TensorRT provider dependencies

## Notes

- Non-JSON human output and JSON machine output are intentionally separated.
- Progress streams are opt-in to keep stdout stable for scripting.
