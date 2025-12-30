# ONNX Export Tooling

This tool exports the individual components of a Parakeet-TDT `.nemo` model to ONNX format. These artifacts are required for building TensorRT engines.

## Components Exported
1.  **Encoder**: Audio signal processing.
2.  **Predictor**: Token/Language model part.
3.  **Joint**: Combines encoder and predictor outputs to produce logits and durations (TDT spec).

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

## Expected Shapes (Example for 0.6B)
- **Encoder**: 
  - Input: `[1, 80, T]` (Log-Mel features)
  - Output: `[1, D_enc, T_enc]`
- **Predictor**:
  - Input: `[1, 1]` (Single token for streaming)
  - Output: `[1, D_pred, 1]`
- **Joint**:
  - Inputs: `[1, D_enc, 1]`, `[1, D_pred, 1]`
  - Outputs: `logits [1, 1, 1, V]`, `durations [1, 1, 1]`

## Notes
- This repo uses **torch 2.9+**, where `torch.onnx.export` defaults to **`dynamo=True`** (new `torch.export`-based path).
  - That path can fail on NeMo modules (FakeTensor `data_ptr` / LSTM flatten) and on NeMo code that does `if self.dropout:`.
  - `export.py` **forces the legacy exporter** via `torch.onnx.export(dynamo=False, fallback=False)` and logs this explicitly.

- **Export-only dropout patch**:
  - During predictor/joint export, `export.py` temporarily patches any module attribute literally named `.dropout` where the value is `nn.Dropout`, setting it to `None`.
  - This avoids TorchScript / torch.export failures for patterns like `if self.dropout: x = self.dropout(x)`.
  - The patch is scoped to the predictor/joint path and is restored afterwards; it is logged.

- Dynamic axes are enabled for the time/token dimensions unless `--fixed` is set.
- The export writes `model_meta.json`, `vocab.txt` (if available), and best-effort `tokenizer.model`.

## Validation
After each export, `export.py` runs:
- `onnx.load_model(..., load_external_data=False)` (for printing IO)
- `onnx.checker.check_model(path)` (validation)
