# TDT Trace Parity Report

Status: FAIL (first divergence at step 0 after engine rebuilds).

## Environment
- Model: `models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo`
- Runtime: `rust/target/debug/cli` + TRT engines (FP16 + FP32 rebuilds)
- Device: CPU (PyTorch), TRT runtime for C++

## Inputs
- WAV: `eval/wav/librispeech_dev_gate/dev-clean/1272-128104-0000.wav`
- Streaming sim: 0.5s (`--stream-sim 0.5`) → 48 frames/chunk
- Feature norm: `per_feature`
- Punct suppression disabled: `PARAKEET_DISABLE_PUNCT_SUPPRESSION=1`
- Debug steps: `PARAKEET_DEBUG_TDT_STEPS=80`

## Commands
```bash
python tools/build_trt/build_trt.py \
  --meta tools/export_onnx/out/model_meta.json \
  --outdir /home/emmy/git/trt-asr-engine/models/parakeet-tdt-0.6b-v3/engines_20260107_logits \
  --fp16

python tools/build_trt/build_trt.py \
  --meta tools/export_onnx/out/model_meta.json \
  --outdir /home/emmy/git/trt-asr-engine/models/parakeet-tdt-0.6b-v3/engines_20260107_fp32

LD_LIBRARY_PATH=/home/emmy/git/trt-asr-engine/cpp/build \
PARAKEET_DEBUG_TDT_STEPS=80 \
PARAKEET_DISABLE_PUNCT_SUPPRESSION=1 \
PARAKEET_FEATURE_NORM=per_feature \
rust/target/debug/cli \
  --model-dir /home/emmy/git/trt-asr-engine/models/parakeet-tdt-0.6b-v3/engines_20260107_fp32 \
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
  - C++ (FP16 rebuilt): `0`
  - C++ (FP32 rebuilt): `0`
- `y_id` matches (`64`, language token), chunk boundaries aligned.
- Logsumexp check (after rebuild):
  - PyTorch `tok_lse=-3.67`, `dur_lse=5.37`
  - C++ `tok_lse≈-3.77`, `dur_lse≈5.80`
  - This indicates **logits scale alignment**, not log‑softmax.

## Interpretation
The mismatch appears **before** decode policy heuristics can explain it and persists after
rebuilding FP16 and FP32 engines from the current logits ONNX exports. Logsumexp alignment
now indicates **matching logits scale**, so the remaining mismatch is likely **numerical drift**
between TRT encoder/joint outputs vs PyTorch (argmax flip on a close duration pair).

## Next Actions
- Run **encoder parity** on the dumped feature chunk (PyTorch vs ORT vs TRT) to localize the argmax flip.
- If encoder parity is tight, add a **duration‑head margin diagnostic** and consider tolerating swaps
  when top‑1/top‑2 are within a small epsilon, or switch to ORT for correctness baselines.

## Step‑0 Snapshot Triage (ORT joint on swapped inputs)
- Command:
  ```bash
  python tools/onnxruntime/compare_joint_step0.py \
    --onnx tools/export_onnx/out/joint.onnx \
    --trt-dir /tmp/tdt_snapshot_trt \
    --pt-dir /tmp/tdt_snapshot_pt
  ```
- Results:
  - `enc_pt + pred_pt` → best_dur_idx=1 (PyTorch baseline)
  - `enc_trt + pred_trt` → best_dur_idx=0
  - `enc_trt + pred_pt` → best_dur_idx=0
  - `enc_pt + pred_trt` → best_dur_idx=1
- Interpretation: **encoder output drift is the root cause** (predictor matches; joint behaves consistently under ORT when fed the same inputs).
