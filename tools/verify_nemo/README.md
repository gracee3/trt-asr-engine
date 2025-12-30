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

### Requirements
- Input WAV should be 16kHz, mono, 16-bit PCM for best results.
- CUDA is recommended but the script will fallback to CPU if unavailable.
