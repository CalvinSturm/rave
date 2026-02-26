# `rave-runtime-nvidia`

Concrete NVIDIA runtime composition for RAVE.

This crate is the opinionated integration layer that wires together:

- `rave-cuda` (CUDA kernels and stream helpers)
- `rave-tensorrt` (TensorRT inference backend)
- `rave-ffmpeg` (container probe/demux/mux)
- `rave-nvcodec` (NVDEC/NVENC)

## Scope

`rave-runtime-nvidia` provides concrete runtime setup helpers used by applications
and the CLI, including:

- input probing and codec/geometry resolution
- CUDA context + kernel compilation
- TensorRT backend initialization
- NVDEC decoder creation
- NVENC encoder creation

## Relationship to `rave-pipeline`

`rave-pipeline` is the generic orchestration crate (stage graph, reports,
pipeline execution, metrics, validation contracts).

`rave-runtime-nvidia` is the concrete backend composition crate.

If you are building an app on the default NVIDIA stack, depend on
`rave-runtime-nvidia` alongside `rave-pipeline`.
