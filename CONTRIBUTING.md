# Contributing

Thanks for your interest in contributing to **trt-asr-engine**.

This repo is a **native, offline, streaming ASR engine prototype** built around NVIDIA Parakeet-TDT and TensorRT. Contributions are welcome, but we want to keep the surface area tight and the behavior reproducible.

## Development environment

- **Linux** is the primary target (CUDA + TensorRT tooling).
- **Rust** (stable toolchain).
- **C++** toolchain + **CMake**.
- **Python 3.10+** for NeMo verification and ONNX export tooling.

## Style expectations

- **Rust**: `cargo fmt` and `cargo clippy` should be clean for touched crates.
- **C++**: keep changes small, readable, and consistent with existing style.
- Prefer **explicit interfaces** over “magic” coupling (especially for predictor/joint tensor layouts).

## Running the Python tools

- **NeMo verification**: see `tools/verify_nemo/README.md`
- **ONNX export**: see `tools/export_onnx/README.md`

## Do not commit model artifacts

Model weights and derived artifacts are **local-only** and must not be committed:

- `models/**` (downloaded `.nemo` weights)
- `tools/export_onnx/out/**` (exported ONNX + external data)
- `tools/build_trt/out/**` and `*.engine` (hardware-specific TensorRT engines)

If you need to add a new required local artifact, update `.gitignore` and document it in the root `README.md` (or `models/README.md`).

## Discussing changes

- Small fixes: open a PR.
- Larger design changes (interfaces, decoding strategy, engine formats): open an issue first to align on direction.


