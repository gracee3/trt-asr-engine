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
python export.py --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo --out ./out
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
- `opset_version=17` is used for compatibility.
- Dynamic axes are enabled for the time/token dimensions.
- The `model_meta.json` file contains important feature extraction parameters (n_mels, sample_rate, etc.).
