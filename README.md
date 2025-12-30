# Parakeet TRT STT Prototype

This repository contains a standalone prototype for high-performance Speech-to-Text using NVIDIA's Parakeet-TDT model, optimized with TensorRT and a C++ runtime.

## Architecture
Rust features → ONNX export → TensorRT engines → C++ runtime (C ABI) → Rust FFI → CLI

## Project Structure
- `models/`: Local model assets (gitignored).
- `tools/verify_nemo/`: Python harness to validate model correctness using NeMo.
- `tools/export_onnx/`: Tooling to export .nemo models to ONNX components.
- `tools/build_trt/`: Scripts to build TensorRT engines from ONNX.
- `cpp/`: TensorRT runtime with a stable C ABI.
- `rust/`: Audio feature extraction, FFI bindings, and CLI.

## Getting Started

### 1. Model Preparation
The model used is [NVIDIA Parakeet-TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
Place the `.nemo` file in `models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo`.

### 2. NeMo Verification
Validate the model locally:
```bash
cd tools/verify_nemo
pip install -r requirements.txt
python verify.py --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo --wav path/to/audio.wav
```

### 3. ONNX Export
Export the encoder, predictor, and joint components:
```bash
cd tools/export_onnx
pip install -r requirements.txt
python export.py --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo
```

## Attribution
This project uses the Parakeet-TDT 0.6b-v3 model by NVIDIA, licensed under [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
