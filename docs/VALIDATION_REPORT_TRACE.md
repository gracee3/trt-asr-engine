# TDT Trace Parity Report

Status: FAIL (first divergence at step 0 after feature layout fix).

## Environment
- Model: `models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo`
- Runtime: `rust/target/debug/cli` + TRT engines in `models/parakeet-tdt-0.6b-v3`
- Device: CPU (PyTorch), TRT runtime for C++

## Inputs
- WAV: `eval/wav/librispeech_dev_gate/dev-clean/1272-128104-0000.wav`
- Streaming sim: 0.5s (`--stream-sim 0.5`) → 48 frames/chunk
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
- Step 0 (`best_dur_idx`):
  - PyTorch: `1`
  - C++: `0`
- `y_id` matches (`64`, language token), chunk boundaries aligned.
- Logsumexp check:
  - PyTorch `tok_lse=-3.67`, `dur_lse=5.37` (raw logits)
  - C++ `tok_lse=-9.56`, `dur_lse≈0` (duration head appears log‑softmaxed or stale engine)

## Interpretation
The mismatch appears **before** decode policy heuristics can explain it and persists with
aligned chunking + priming. The logsumexp disparity strongly suggests a **stale TRT joint engine**
(built from a log‑softmax export) or a joint graph mismatch, rather than a decode policy bug.

## Next Actions
- Rebuild **joint** (and likely predictor) TRT engines from the latest ONNX exports (logits output).
- Re-run the same 1‑utterance trace; confirm `tok_lse`/`dur_lse` match PyTorch scale and `best_dur_idx` aligns.
