# TDT Trace Parity Report

Status: FAIL (first divergence at step 4).

## Environment
- Model: `models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo`
- Runtime: `rust/target/debug/cli` + TRT engines in `models/parakeet-tdt-0.6b-v3`
- Device: CPU (PyTorch), TRT runtime for C++

## Inputs
- WAV: `eval/wav/librispeech_dev_gate/dev-clean/1272-128104-0000.wav`
- Streaming sim: 0.5s (`--stream-sim 0.5`)
- Feature norm: `per_feature`
- Punct suppression disabled: `PARAKEET_DISABLE_PUNCT_SUPPRESSION=1`
- Debug steps: `PARAKEET_DEBUG_TDT_STEPS=80`

## Commands
```bash
LD_LIBRARY_PATH=/home/emmy/git/trt-asr-engine/cpp/build \
PARAKEET_DEBUG_TDT_STEPS=80 \
PARAKEET_DISABLE_PUNCT_SUPPRESSION=1 \
PARAKEET_FEATURE_NORM=per_feature \
rust/target/debug/cli \
  --model-dir models/parakeet-tdt-0.6b-v3 \
  --stream-sim 0.5 \
  --dump-features /tmp/tdt_features.f32 \
  eval/wav/librispeech_dev_gate/dev-clean/1272-128104-0000.wav \
  > /tmp/cpp_tdt_stdout.log \
  2> /tmp/cpp_tdt_stderr.log

python tools/verify_nemo/tdt_trace.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --model-dir models/parakeet-tdt-0.6b-v3 \
  --features-f32 /tmp/tdt_features.f32 \
  --chunk-frames 48 \
  --out /tmp/pt_tdt_trace.jsonl \
  --device cpu \
  --contract contracts/parakeet-tdt-0.6b-v3.contract.json \
  --max-steps 80

python tools/verify_nemo/compare_tdt_trace.py \
  --pt-trace /tmp/pt_tdt_trace.jsonl \
  --cpp-trace /home/emmy/git/trt-asr-engine/.cursor/debug.log \
  --max-steps 80 \
  --check-index
```

## First Divergence
- Step 4 (`best_tok`):
  - PyTorch: `8192` (blank)
  - C++: `7883`
- Token head top‑k (PyTorch): blank is top‑1; C++ top‑1 is non‑blank with lower absolute logits.
- Duration head scale differs (PyTorch positive logits vs C++ negative logits).

## Interpretation
The mismatch appears **before** decode policy heuristics can explain it and is consistent with
**encoder output / feature pipeline differences** or **engine mismatch** (TRT encoder built
from a different export than the current PyTorch model).

## Next Actions
- Run offline encoder parity (PyTorch vs TRT/ORT) using the same dumped features.
- Ensure encoder engine is rebuilt from the latest ONNX export aligned to the current contract.
