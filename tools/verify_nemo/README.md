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

### Model architecture audit (FastConformer/TDT checks)
```bash
python audit_model_arch.py \
  --model ../../models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out audit_model_arch.json
```

### Requirements
- Input WAV should be 16kHz, mono, 16-bit PCM for best results.
- CUDA is recommended but the script will fallback to CPU if unavailable.
