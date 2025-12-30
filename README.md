# Parakeet TDT TensorRT Prototype

Offline, true streaming ASR prototype using Rust for feature extraction and C++ for TensorRT inference.

## Objective
Prove the viability of a high-performance streaming ASR pipeline:
**Rust Audio (CPU) -> TensorRT (FP16/GPU) -> Rust FFI -> CLI**

## Model Attribution
This project uses the **Parakeet-TDT 0.6B v3** model by NVIDIA.
- **Model Card:** [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- **License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

## Project Structure
- `tools/export_onnx/`: Python scripts to export NeMo models to ONNX.
- `tools/build_trt/`: Scripts to compile ONNX to TensorRT engines.
- `cpp/`: TensorRT runtime implementation with a stable C ABI.
- `rust/`: Workspace containing:
    - `features`: Log-mel feature extraction.
    - `parakeet_trt_sys`: Low-level FFI bindings.
    - `parakeet_trt`: High-level safe wrapper.
    - `cli`: Command-line transcription tool.

## Requirements
- CUDA 12.x + cuDNN
- TensorRT 10.x
- Rust 1.75+
- CMake 3.20+
- SentencePiece (C++ library)

## Installation & Build
```bash
# 1. Build C++ Runtime
cd cpp && mkdir build && cd build
cmake ..
make -j$(nproc)

# 2. Build Rust CLI
cd ../../rust
cargo build --release
```

## Usage
Expects a model directory with:
- `encoder.engine`
- `predictor.engine`
- `joint.engine`
- `tokenizer.model`
- `model_meta.json`

```bash
cargo run --bin transcribe -- <wav_file> --model-dir ./models/parakeet-tdt-0.6b-v3 --stream-sim 0.5
```

## Verification Plan
1. Use `tools/export_onnx/golden_transcript.py` to get the reference output from NeMo.
2. Run the CLI and compare the result.
3. Validate feature extractor output against NeMo's `AudioToMelSpectrogram`.
