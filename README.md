# Parakeet TensorRT Streaming STT (Prototype)

This repository is a **standalone prototype** for **offline, low-latency speech-to-text** using:

* **NVIDIA Parakeet-TDT-0.6B-v3**
* **Rust audio preprocessing / feature extraction**
* **TensorRT inference (FP16)**
* **C++ runtime with C ABI**
* **Rust FFI + CLI demo**

The goal is to validate a **true streaming ASR backend** (no chunk overlap hacks) before integrating into the larger Talisman system.

This repo intentionally starts *clean* and *minimal*.

---

## Status

**Phase 1 (current):**

* Offline model usage
* Rust feature extraction (log-mel fbank)
* TensorRT inference via C++ runtime
* Rust FFI wrapper
* CLI demo (WAV transcription + simulated streaming)

**Out of scope (by design):**

* Microphone capture
* UI / Patch Bay integration
* Exposing STT as a PipeWire node
* GPU feature extraction
* Beam search decoding (greedy only for now)

---

## Model

We use:

```
nvidia/parakeet-tdt-0.6b-v3
```

### Why Parakeet TDT

* **True streaming** transducer (TDT / RNN-T family)
* Very high RTFx (real-time factor)
* Low latency
* Open license (CC-BY-4.0)
* Designed for production ASR

### License

This model is licensed under **CC-BY-4.0**.

You **must provide attribution** in any redistributed software or documentation.

Example attribution:

> “This product uses the Parakeet-TDT-0.6B-v3 speech recognition model by NVIDIA, licensed under CC-BY-4.0.”

Model card:
[https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

---

## Repository Layout

```
parakeet-trt-stt/
  README.md

  models/                        # local model cache (not committed)
    parakeet-tdt-0.6b-v3/
      encoder.engine
      predictor.engine
      joint.engine
      tokenizer.model / vocab.txt
      model_meta.json

  tools/
    export_onnx/
      export.py                  # NeMo → ONNX exporter
      requirements.txt
      README.md

    build_trt/
      build.sh                   # ONNX → TensorRT FP16 engines
      README.md

  cpp/
    CMakeLists.txt
    include/
      parakeet_trt.h             # C ABI header
    src/
      parakeet_trt.cpp           # C ABI implementation
      trt_runner.cpp             # TensorRT engine wrapper
      decoder.cpp                # Streaming transducer decode (greedy)
      tokenizer.cpp              # SentencePiece decoding
      utils.cpp

  rust/
    Cargo.toml                   # workspace
    crates/
      features/                  # Rust fbank/log-mel extractor
      parakeet_trt_sys/          # bindgen + raw FFI
      parakeet_trt/              # safe Rust wrapper
      cli/                       # demo CLI
```

---

## Audio Input Requirements

The STT engine **expects**:

* **16 kHz**
* **Mono**
* **PCM float (`f32`)** internally

The CLI will:

* load WAV
* resample if needed
* convert to mono
* feed features incrementally

---

## Feature Extraction (Rust)

Features are computed **in Rust (CPU)** and fed to TensorRT.

### Parameters (match NeMo training defaults)

| Parameter     | Value                   |
| ------------- | ----------------------- |
| Sample rate   | 16,000 Hz               |
| Window length | 25 ms                   |
| Hop length    | 10 ms                   |
| FFT size      | 512                     |
| Mel bins      | 80                      |
| Compression   | log / log1p             |
| Normalization | optional (configurable) |

### Output Shape

```
[num_frames, 80]
```

Frames are passed incrementally to the encoder engine.

> ⚠️ Exact normalization parameters must match the NeMo config.
> A Python verification harness is provided to validate correctness.

---

## TensorRT Engines

We use **three TensorRT engines**, built offline:

1. **Encoder**
2. **Predictor**
3. **Joint**

All engines are:

* **FP16**
* Built for **batch size = 1**
* Dynamic in time dimension

### Expected files in `model_dir`

```
encoder.engine
predictor.engine
joint.engine
tokenizer.model (or vocab.txt)
model_meta.json
```

---

## `model_meta.json` (required)

Example:

```json
{
  "sample_rate": 16000,
  "feature_dim": 80,
  "blank_id": 0,
  "vocab_size": 8192,
  "model": "parakeet-tdt-0.6b-v3",
  "frontend": {
    "type": "fbank",
    "win_length_ms": 25,
    "hop_length_ms": 10,
    "n_fft": 512,
    "n_mels": 80
  }
}
```

This file allows the runtime to stay decoupled from training configs.

---

## Building the Engines (Offline)

### 1) Export ONNX (Python / NeMo)

In `tools/export_onnx/`:

```bash
pip install -r requirements.txt
python export.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --output onnx/
```

Expected outputs:

```
encoder.onnx
predictor.onnx
joint.onnx
```

> If export is incomplete, this directory documents the expected shapes and naming.
> ONNX export is the **highest-risk step** in this project.

---

### 2) Build TensorRT Engines

In `tools/build_trt/`:

```bash
./build.sh onnx/ models/parakeet-tdt-0.6b-v3/
```

Uses `trtexec` to generate FP16 engines.

---

## C++ Runtime (TensorRT)

The C++ runtime:

* loads all three engines
* runs streaming inference
* implements greedy transducer decoding
* emits incremental text events

### C ABI (simplified)

```c
ParakeetHandle* parakeet_create(const char* model_dir, int device_id);
void parakeet_destroy(ParakeetHandle*);

int parakeet_push_features(
  ParakeetHandle*,
  const float* features,
  int frames,
  int feat_dim,
  int end_of_utterance
);

int parakeet_poll_events(
  ParakeetHandle*,
  ParakeetEvent* out_events,
  int max_events
);
```

Events:

* `PARTIAL_TEXT`
* `FINAL_TEXT`
* `ERROR`

---

## Rust FFI + API

Rust exposes a safe wrapper:

```rust
let mut stt = ParakeetSession::new(model_dir, device_id)?;
let events = stt.push_features(&features, end);
```

Text output is structured as:

```rust
enum TextEvent {
    Partial { segment_id, text },
    Final { segment_id, text },
    Error(String),
}
```

---

## CLI Demo

### Build

```bash
cargo build --release
```

### Transcribe WAV

```bash
cargo run -p cli -- \
  transcribe path/to/audio.wav \
  --model-dir models/parakeet-tdt-0.6b-v3
```

### Simulated streaming (partial updates)

```bash
cargo run -p cli -- \
  transcribe path/to/audio.wav \
  --model-dir models/parakeet-tdt-0.6b-v3 \
  --stream-sim 0.2
```

---

## Python Verification Harness (Strongly Recommended)

Before trusting TensorRT results, validate **frontend + transcript** using NeMo:

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v3"
)

out = model.transcribe(["test.wav"], timestamps=True)
print(out[0].text)
```

Use this output as the **golden reference**.

---

## Performance Notes

* RTX 5000 (16GB) is more than sufficient.
* Expect:

  * very low latency
  * real streaming behavior
  * headroom for concurrent LLM tasks (summarization, etc.)

GPU scheduling and priority control will matter once integrated into a larger system.

---

## Attribution (Required)

This software uses:

* **Parakeet-TDT-0.6B-v3** by NVIDIA
  Licensed under **CC-BY-4.0**
  [https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

---

## Why This Repo Exists

This prototype exists to answer one question definitively:

> *Can we run a true streaming, low-latency, offline ASR engine locally, with full control over performance and integration?*

Once validated, this code will be **integrated into Talisman** as a first-class STT backend.

---

## Next Steps (After Phase 1)

* Add beam search
* Add VAD-driven endpointing
* GPU feature extraction
* PipeWire audio capture
* Patch Bay integration
* Summary / LLM modules


