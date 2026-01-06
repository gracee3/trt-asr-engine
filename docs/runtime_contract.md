# Runtime tensor contract (Parakeet TDT 0.6B v3, TensorRT)

This document **locks the runtime tensor contract** for the TensorRT engines exported/built from:

- `tools/export_onnx/out/{encoder,predictor,joint}.onnx`
- `tools/export_onnx/out/model_meta.json`

If you change any tensor names, ranks, layouts, or dtypes after this point, **treat it as a breaking change** for:

- C++ runtime (`cpp/`)
- Rust FFI (`rust/parakeet_trt_sys`, `rust/parakeet_trt`)
- Any prebuilt TensorRT engines (`*.engine`)

## Global assumptions (Phase 2 lock)

- **Batch**: \(B=1\) only (all profiles and runtime code assume batch 1).
- **Precision**: FP16 everywhere feasible (TensorRT `--fp16` build), but some tensors (lengths / token ids) are **INT64**.
- **Layout**: channels-first for encoder features: `[B, 128, T]`.
- **Stream safety**: runtime uses a per-session CUDA stream; device buffers are allocated once and reused.
- **Bindings**: runtime must resolve bindings by **name** (not hardcoded indices). Indices can differ across TensorRT versions/builds.

## Engine: `encoder.engine`

### Bindings (names)

These are the **exact ONNX graph IO names** and are expected to be preserved as TensorRT binding names:

- **Inputs**
  - `audio_signal`
  - `length`
- **Outputs**
  - `encoder_output`
  - `encoded_lengths`

### Shapes, dtypes, layouts

- **`audio_signal`**: `FLOAT` (FP16 at runtime when using FP16 engines)
  - Shape: `[B, 128, T]`
  - Layout: `BCT` (batch, mel, time)
- **`length`**: `INT64`
  - Shape: `[1]` (rank-1)
  - Meaning: valid `T` for `audio_signal` (frames)
- **`encoder_output`**: `FLOAT` (FP16 at runtime when using FP16 engines)
  - Shape: `[B, 1024, T_enc]`
  - Layout: `BCT`
- **`encoded_lengths`**: `INT64`
  - Shape: `[1]`

### Shape rules

- \(B=1\)
- \(T\) is dynamic and must be within the encoder optimization profile.
- \(T_{enc}\) is model-dependent (subsampled relative to \(T\)); do not assume a fixed ratio in runtime code. Always read `encoded_lengths`.

## Engine: `predictor.engine`

### Bindings (names)

- **Inputs**
  - `y`
  - `h`
  - `c`
- **Outputs**
  - `g`
  - `h_out`
  - `c_out`

### Shapes, dtypes, layouts

- **`y`**: `INT64`
  - Shape: `[B, U]`
  - Meaning: token IDs (for greedy decode, typically `U=1` per step)
- **`h` / `c`**: `FLOAT` (FP16 at runtime when using FP16 engines)
  - Shape: `[L, B, H]` where:
    - \(L=2\) (num layers)
    - \(H=640\) (hidden)
- **`g`**: `FLOAT` (FP16 at runtime when using FP16 engines)
  - Shape: `[B, H, U]`
  - Layout: `BHU`
  - Note: this is **transposed** vs some NeMo implementations which use `[B, U, H]`.
- **`h_out` / `c_out`**: `FLOAT` (FP16 at runtime when using FP16 engines)
  - Shape: `[L, B, H]`

### Shape rules

- \(B=1\)
- \(U\) is dynamic and must be within the predictor optimization profile; greedy decode uses `U=1`.
- Predictor state (`h`/`c`) is carried across decode steps and must be initialized to zeros at utterance start.

## Engine: `joint.engine`

### Bindings (names)

- **Inputs**
  - `encoder_output`
  - `predictor_output`
- **Outputs**
  - `joint_output`

### Shapes, dtypes, layouts

- **`encoder_output`**: `FLOAT` (FP16 at runtime when using FP16 engines)
  - Shape: `[B, 1024, T]`
  - Layout: `BCT`
- **`predictor_output`**: `FLOAT` (FP16 at runtime when using FP16 engines)
  - Shape: `[B, 640, U]`
  - Layout: `BHU` (matches predictor `g`)
- **`joint_output`**: `FLOAT` (FP16 at runtime when using FP16 engines)
  - Shape: `[B, T, U, V]`
  - Layout: `BTUV`
  - \(V=8198\) (from `model_meta.json` `joint_vocab_size`)
  - Output is raw logits (no log-softmax); token and duration heads must be normalized independently if probabilities are required.

### Shape rules

- \(B=1\)
- Greedy decode typically uses `T=1, U=1` for step-wise joint evaluation.
- \(T\) and \(U\) must be within the joint optimization profile.

## Optimization-profile “opt” smoke-test shapes (Phase 2)

When building engines via `tools/build_trt/build_trt.py` defaults, the **opt shapes** are:

- **Encoder**
  - `audio_signal`: `1x128x64`
  - `length`: `1`
- **Predictor**
  - `y`: `1x1`
  - `h`: `2x1x640`
  - `c`: `2x1x640`
- **Joint**
  - `encoder_output`: `1x1024x64`
  - `predictor_output`: `1x640x1`

## Binding map (name → index → shape rule)

### Rule: do not hardcode indices

TensorRT binding indices **can vary** across versions/build flags. Runtime must map by name:

- `engine->getBindingIndex("audio_signal")`, etc.

### Capturing the actual indices for a specific build

After engines are built, record the exact indices by running:

```bash
python tools/build_trt/scripts/inspect_engine.py --engine models/parakeet-tdt-0.6b-v3/encoder.engine --shapes audio_signal:1x128x64,length:1
python tools/build_trt/scripts/inspect_engine.py --engine models/parakeet-tdt-0.6b-v3/predictor.engine --shapes y:1x1,h:2x1x640,c:2x1x640
python tools/build_trt/scripts/inspect_engine.py --engine models/parakeet-tdt-0.6b-v3/joint.engine --shapes encoder_output:1x1024x64,predictor_output:1x640x1
```

Then copy the printed binding table (or add a small script to parse `trtexec --verbose`) into this section and treat it as frozen for that engine build.

