# trt-asr-engine

**trt-asr-engine** is an **offline, native, streaming ASR engine prototype** for NVIDIA **Parakeet-TDT-0.6B-v3**, with:

- **Rust** audio feature extraction
- **ONNX** component export (encoder / predictor / joint)
- **TensorRT FP16** inference
- **C++ runtime** with a stable **C ABI**
- **Rust FFI** + a small **CLI demo**

This repository focuses on the **engine/runtime** building blocks and reproducible artifacts. It is not an end-user speech app.

## What this is (and what it is not)

- **This is**:
  - offline-first (local model weights)
  - streaming-native (transducer-style components + recurrent predictor state)
  - optimized for TensorRT deployment (FP16 engines)
  - suitable as a component inside larger systems

- **This is not**:
  - a hosted/cloud API
  - a GUI application
  - a full microphone capture application (yet)
  - a redistribution of model weights

## Architecture overview

High-level pipeline:

```
Audio
  → Rust features (log-mel)
  → ONNX (encoder / predictor / joint)
  → TensorRT engines (FP16)
  → C++ runtime (C ABI)
  → Rust FFI
  → CLI demo
```

Why TensorRT + transducer-style models:

- **TensorRT** provides strong latency/throughput for deployment on NVIDIA GPUs.
- **RNNT/TDT-style transducers** map naturally to streaming: you can incrementally run the encoder on new audio and step the predictor/joint during decode.

## Quick start (developer-oriented)

### Prerequisites (development)

- Linux + NVIDIA GPU (recommended)
- CUDA + TensorRT installed for engine build steps
- Rust toolchain
- CMake + a C++ compiler
- Python 3.10+ (for NeMo verification + ONNX export tooling)

### 1) Create a Python environment (tooling)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r tools/verify_nemo/requirements.txt
pip install -r tools/export_onnx/requirements.txt
```

### 2) Download the model weights (local-only)

Model card: [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

This repo does **not** ship model weights. Put the `.nemo` file here:

`models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo`

See `models/README.md` for details.

### 3) Verify the `.nemo` locally (NeMo)

```bash
python -u tools/verify_nemo/verify.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --wav path/to/audio.wav
```

### 4) Export ONNX components (encoder / predictor / joint)

```bash
python -u tools/export_onnx/export.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out tools/export_onnx/out \
  --component all \
  --device cpu \
  --smoke-test-ort
```

This produces:

- `tools/export_onnx/out/encoder.onnx` (may also emit `encoder.onnx.data`)
- `tools/export_onnx/out/predictor.onnx`
- `tools/export_onnx/out/joint.onnx`
- `tools/export_onnx/out/model_meta.json` (includes tensor layout contracts)
- tokenizer/vocab assets

### 5) Build TensorRT engines (placeholder)

Engine build scripts live under `tools/build_trt/` and require **TensorRT’s `trtexec`** to be installed and available in `PATH`.

Minimal command:

```bash
python tools/build_trt/build_trt.py \
  --meta tools/export_onnx/out/model_meta.json \
  --outdir models/parakeet-tdt-0.6b-v3 \
  --fp16
```

Expected outputs:

- `models/parakeet-tdt-0.6b-v3/{encoder,predictor,joint}.engine`
- `models/parakeet-tdt-0.6b-v3/build_report.json`
- `models/parakeet-tdt-0.6b-v3/build_logs/*.log`

### 6) Build the C++ runtime (libparakeet_trt.so)

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
```

The shared library will be at `cpp/build/libparakeet_trt.so`.

## Repository structure

- **`tools/`**: dev tooling pipeline
  - `tools/verify_nemo/`: NeMo verification harness (golden output)
  - `tools/export_onnx/`: deterministic NeMo → ONNX export (validated)
  - `tools/build_trt/`: TensorRT engine build scripts (WIP)
  - `tools/analyze_tap.py`: audio/feature tap analysis tool
- **`models/`**: local-only `.nemo` weights (gitignored; see `models/README.md`)
- **`cpp/`**: C++ TensorRT runtime + C ABI
  - `cpp/include/audio_tap.h`: reusable audio tap writer for pipeline debugging
- **`rust/`**: feature extraction, FFI bindings, CLI demo
- **`docs/`**: documentation
  - `docs/debugging.md`: comprehensive debugging guide
  - `docs/runtime_contract.md`: tensor interface contracts

## Debugging integration issues

See **[docs/debugging.md](docs/debugging.md)** for comprehensive debugging documentation.

### Quick reference: environment variables

```bash
# Audio taps (for Magnolia integration)
AUDIO_TAP_ENABLE=1          # Enable all audio taps
AUDIO_TAP_FEATURES=1        # Enable feature tap in trt-asr-engine

# NaN guards
PARAKEET_NAN_GUARD_ALWAYS=1 # Check every chunk for NaN/Inf
PARAKEET_NAN_GUARD_HALT=1   # Abort on first NaN/Inf

# Cache debugging
PARAKEET_CACHE_LEN_OVERRIDE=-1  # Use cache capacity as cache_len
```

### Replay harness

The Rust CLI supports replaying captured audio/features for deterministic reproduction:

```bash
# Replay WAV file
./target/debug/cli test.wav --model-dir ./models/parakeet-tdt-0.6b-v3 -v

# Replay raw PCM tap
./target/debug/cli tap_post_dsp.raw --raw-pcm --model-dir ./models/... -v

# Replay feature tap (bypass feature extraction)
./target/debug/cli tap_features.raw --features-input --model-dir ./models/... -v
```

## Licensing & attribution

- **Code license**: Apache-2.0 (see `LICENSE`)
- **Model license**: NVIDIA Parakeet-TDT-0.6B-v3 is licensed separately under **CC-BY-4.0**:
  - Model card: [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  - You are responsible for complying with NVIDIA’s attribution requirements when using the model.

This repository **does not redistribute model weights**.

## Status / roadmap

- **Phase 1 (complete)**: deterministic NeMo → ONNX export with validated predictor/joint interfaces
- **Next**:
  - TensorRT engine build + packaging (including external ONNX data)
  - streaming C++ decode loop integration
  - Rust CLI polish and integration into larger systems
