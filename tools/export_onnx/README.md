# ONNX Export Tooling

This tool exports the individual components of a Parakeet-TDT `.nemo` model to ONNX format. These artifacts are required for building TensorRT engines.

## Components Exported
1.  **Encoder**: Audio signal processing.
2.  **Predictor**: Token/Language model part.
3.  **Joint**: Combines encoder and predictor outputs to produce logits and durations (TDT spec).
4.  **Streaming Encoder (cache-aware)**: Encoder with explicit cache inputs/outputs for true streaming.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python -u export.py \
  --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out ./out \
  --component all \
  --device cpu
```

To export only the predictor:

```bash
python -u export.py \
  --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out ./out \
  --component predictor \
  --device cpu
```

To add a minimal ONNX Runtime smoke test (recommended for CI):

```bash
python -u export.py \
  --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out ./out \
  --component all \
  --device cpu \
  --smoke-test-ort
```

To export a cache-aware streaming encoder (explicit cache I/O):

```bash
python -u export.py \
  --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out ./out \
  --component encoder_streaming \
  --streaming-cache-size 256 \
  --streaming-cache-drop-size 0 \
  --device cpu
```

## Expected Shapes (Example for 0.6B)
- **Encoder**: 
  - Input: `audio_signal [B, 128, T]` and `length [1]` (**as exported**)
  - Outputs: `encoder_output [B, 1024, T_enc]` and `encoded_lengths [1]`
- **Streaming Encoder**:
  - Inputs:
    - `audio_signal [B, 128, T]` and `length [B]`
    - `cache_last_channel [L, B, cache_T, 1024]`
    - `cache_last_time [L, B, 1024, C]`
    - `cache_last_channel_len [B]`
  - Outputs:
    - `encoder_output [B, 1024, T_enc]`
    - `encoded_lengths [B]`
    - `cache_last_channel_out [L, B, cache_T, 1024]`
    - `cache_last_time_out [L, B, 1024, C]`
    - `cache_last_channel_len_out [B]`
- **Predictor**:
  - Inputs:
    - `y`: token IDs `[B, U]` (typically streaming uses `U=1`)
    - `h`: LSTM hidden state `[L, B, H]`
    - `c`: LSTM cell state `[L, B, H]`
  - Outputs:
    - `g`: predictor embedding `[B, H, U]` (this is **transposed** from NeMo’s internal `[B, U, H]` for joint compatibility)
    - `h_out`: next hidden state `[L, B, H]`
    - `c_out`: next cell state `[L, B, H]`
- **Joint**:
  - Inputs:
    - `encoder_output`: `[B, D_enc, T]`
    - `predictor_output`: `[B, H, U]`
  - Output:
    - `joint_output`: `[B, T, U, V_joint]` (for this model `V_joint = 8198`)

## Notes
- This repo uses **torch 2.9+**, where `torch.onnx.export` defaults to **`dynamo=True`** (new `torch.export`-based path).
  - That path can fail on NeMo modules (FakeTensor `data_ptr` / LSTM flatten) and on NeMo code that does `if self.dropout:`.
  - `export.py` **forces the legacy exporter** via `torch.onnx.export(dynamo=False, fallback=False)` and logs this explicitly.

- **Export-only dropout patch**:
  - During predictor/joint export, `export.py` temporarily patches any module attribute literally named `.dropout` where the value is `nn.Dropout`, setting it to `None`.
  - This avoids TorchScript / torch.export failures for patterns like `if self.dropout: x = self.dropout(x)`.
  - The patch is scoped to the predictor/joint path and is restored afterwards; it is logged.

- **Export-only joint fuse patch**:
  - Some NeMo versions ship `RNNTJoint` with `fuse_loss_wer=True`, which forces transcript/length inputs (training plumbing).
  - `export.py` temporarily disables this for export so the joint graph is pure inference: `(encoder_output, predictor_output) -> joint_output`.

- **Encoder external data (`.onnx.data`)**:
  - If the encoder ONNX exceeds protobuf size limits, PyTorch will emit `encoder.onnx` **plus** `encoder.onnx.data` (external weights).
  - Keep these together for TensorRT build steps.

- Dynamic axes are enabled for the time/token dimensions unless `--fixed` is set.
- The export writes `model_meta.json`, `vocab.txt` (if available), and best-effort `tokenizer.model`.
- **Joint output is raw logits** (no log-softmax). Runtime should split token vs duration heads and apply per-head softmax only when probabilities are required.

## Validation
After each export, `export.py` runs:
- `onnx.load_model(..., load_external_data=False)` (for printing IO)
- `onnx.checker.check_model(path)` (validation)
