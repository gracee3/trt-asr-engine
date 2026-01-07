# NeMo Verification Harness

This tool validates the `.nemo` model file by running a local transcription on a test WAV file. It ensures the model is loaded correctly and produces the expected output ("golden transcript").

## Setup

Ensure you have a Python environment with NeMo installed:
```bash
pip install -r requirements.txt
```

## Usage

Run the verification script:
```bash
python verify.py --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo --wav path/to/sample.wav
```

### Encoder cache-aware streaming sanity check
```bash
python streaming_encoder_cache.py \
  --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --cache-size 256 \
  --num-chunks 3
```
Stateful cache carryover is enabled by default; the script clamps `cache_drop_size` if it would yield negative cache lengths.

### Model architecture audit (FastConformer/TDT checks)
```bash
python audit_model_arch.py \
  --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out audit_model_arch.json
```

### Reference JSONL with streaming schedule
```bash
python streaming_encoder_reference.py \
  --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --device cpu \
  --cache-size 256 \
  --use-streaming-cfg-schedule \
  --num-chunks 50 \
  --jsonl-out artifacts/reference/pytorch_reference_50.jsonl
```

### Requirements
- Input WAV should be 16kHz, mono, 16-bit PCM for best results.
- CUDA is recommended but the script will fallback to CPU if unavailable.
