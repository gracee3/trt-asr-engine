# TDT Trace Parity Report

Status: PARTIAL (ORT streaming cache parity PASS after cache_drop_size=3 re-export; TRT closed-loop parity failing due to cache output deltas).

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

## Streaming Encoder Parity (ORT vs TRT, step‑0)
### Build streaming encoder engine (FP32)
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/emmy/git/trt-asr-engine/tools/export_onnx/out/encoder_streaming.onnx \
  --saveEngine=/home/emmy/git/trt-asr-engine/models/parakeet-tdt-0.6b-v3/engines_20260107_fp32/encoder_streaming.engine \
  --minShapes=audio_signal:1x128x16,length:1,cache_last_channel:1x24x256x1024,cache_last_time:1x24x1024x4,cache_last_channel_len:1 \
  --optShapes=audio_signal:1x128x64,length:1,cache_last_channel:1x24x256x1024,cache_last_time:1x24x1024x4,cache_last_channel_len:1 \
  --maxShapes=audio_signal:1x128x256,length:1,cache_last_channel:1x24x256x1024,cache_last_time:1x24x1024x4,cache_last_channel_len:1
```

### Dump TRT step‑0 snapshot (streaming encoder override)
```bash
LD_LIBRARY_PATH=/home/emmy/git/trt-asr-engine/cpp/build \
PARAKEET_STREAMING_ENCODER_PATH=/home/emmy/git/trt-asr-engine/models/parakeet-tdt-0.6b-v3/engines_20260107_fp32/encoder_streaming.engine \
PARAKEET_DEBUG_TDT_STEPS=1 \
PARAKEET_TDT_SNAPSHOT_DIR=/tmp/tdt_snapshot_trt_stream10 \
rust/target/debug/cli \
  --model-dir /home/emmy/git/trt-asr-engine/models/parakeet-tdt-0.6b-v3/engines_20260107_fp32 \
  --stream-sim 0.5 \
  --device-id 0 \
  --verbose \
  eval/wav/librispeech_dev_gate/dev-clean/1272-128104-0000.wav
```

### Compare ORT encoder vs TRT encoder (step‑0)
```bash
python tools/onnxruntime/compare_encoder_step0.py \
  --trt-dir /tmp/tdt_snapshot_trt_stream10
```

### Results
- `enc_out_t0` parity: `max_abs=3.1e-05`, `mean_abs=5.0e-06`, `p99_abs=1.9e-05`
- Cache inputs at t=0 are zero (`cache_in max_abs: channel=0.0 time=0.0`)
- `cache_last_channel_len_out`:
  - ORT: `-67` (raw output)
  - TRT: raw `-67`, fallback `2`
- `encoder_output` shapes: ORT `[1,1024,2]` vs TRT joint slice `[1,1024,16]` (full‑tensor parity skipped due to shape mismatch)
- Cache outputs remain zero (`cache_out max_abs cache_last_channel_out=0 cache_last_time_out=0`), so cache propagation currently advances only via fallback length (no nonzero cache state).

### Interpretation
- **TRT streaming encoder matches ORT streaming encoder** on step‑0 given identical inputs and zero caches.
- Remaining mismatch with PyTorch trace is **not** a TRT encoder bug; the PyTorch trace is still using offline encoder semantics.
- `cache_last_channel_len_out=-67` is coming from the ONNX export (ORT matches TRT), so cache‑len semantics must be corrected at the export/model level. Runtime now applies a fallback (`cache_len_in + T_enc`) to unblock multi‑chunk runs.

Note: This step‑0 parity reflects the pre‑`cache_drop_size=3` export. Rebuild TRT streaming encoder from the new ONNX and re‑run step‑0 parity to update these values.

## ORT Streaming Parity (4 chunks, closed-loop)
### Reference generation (PyTorch streaming + cache_drop_size=3 override)
```bash
python tools/verify_nemo/streaming_encoder_reference.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --device cpu \
  --chunk-len 48 \
  --num-chunks 4 \
  --cache-drop-size 3 \
  --chunk-size 48 \
  --jsonl-out /tmp/stream_ref_cache3.jsonl
```

### ORT closed-loop parity (cache_drop_size=3 export)
```bash
python tools/onnxruntime/onnx_streaming_parity.py \
  --onnx tools/export_onnx/out/encoder_streaming.onnx \
  --ref /tmp/stream_ref_cache3.jsonl \
  --mode closed_loop \
  --providers cpu \
  --max-chunks 4 \
  --summary-json /tmp/ort_streaming_parity_cache3.json
```

### ORT cache sensitivity (sanity)
```bash
python tools/onnxruntime/ort_cache_sensitivity.py \
  --snapshot-dir /tmp/tdt_snapshot_trt_stream10 \
  --providers cpu
```

### Results
- cache_len_out: `1,2,3,4` (ORT matches reference; monotonic)
- encoded_lengths: `3` for all chunks (ORT matches)
- encoder_output parity: max_abs ≤ `2.8e-07` over 4 chunks
- cache_last_time_out: non-zero; cache_last_channel_out remains zero at these chunks
- ORT cache sensitivity: `cache_len_out A=1 B=65`, cache outputs non-zero and sensitive to cache inputs

### Interpretation
- Closed‑loop parity now **PASS** with stateful cache_len_out after re‑exporting streaming encoder with `cache_drop_size=3`.
- Next: rebuild TRT streaming encoder from the new ONNX and re-run TRT vs ORT closed-loop parity.

## ORT Streaming Parity (50 chunks, cache3 schedule)
### Reference generation (cache3 schedule with pre-encode)
```bash
python tools/verify_nemo/streaming_encoder_reference.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --device cpu \
  --chunk-len 48 \
  --num-chunks 50 \
  --cache-drop-size 3 \
  --chunk-size 48 \
  --shift-size 24 \
  --use-streaming-cfg-schedule \
  --jsonl-out artifacts/reference/stream_ref_cache3_50.jsonl
```

### ORT closed-loop parity (cache-aware tolerances)
```bash
python tools/onnxruntime/onnx_streaming_parity.py \
  --onnx tools/export_onnx/out/encoder_streaming.onnx \
  --ref artifacts/reference/stream_ref_cache3_50.jsonl \
  --mode closed_loop \
  --providers cpu \
  --summary-json artifacts/parity/ort_streaming_parity_cache3_50.json
```

### Results
- PASS `50/50`.
- encoder_output max_abs ≤ `5.476e-07` (per-chunk logs).
- cache_last_channel_out compared on valid region only (cache_len).
- cache_last_time_out compared on valid region only (K inferred from reference; last slot is zero).
- cache_last_channel_len_out monotonic (`1..148`).

### Interpretation
- ORT closed-loop parity confirms export correctness with cache3 schedule under cache-aware tolerances and valid-region masking.

## TRT Streaming Parity (50 chunks, cache3 schedule, T=41/57 profile)
### Engine build (T=41/57 profile)
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/emmy/git/trt-asr-engine/tools/export_onnx/out/encoder_streaming.onnx \
  --saveEngine=/home/emmy/git/trt-asr-engine/models/parakeet-tdt-0.6b-v3/engines_20260108_cache3_fp32_t57/encoder_streaming.engine \
  --minShapes=audio_signal:1x128x41,length:1,cache_last_channel:1x24x256x1024,cache_last_time:1x24x1024x4,cache_last_channel_len:1 \
  --optShapes=audio_signal:1x128x57,length:1,cache_last_channel:1x24x256x1024,cache_last_time:1x24x1024x4,cache_last_channel_len:1 \
  --maxShapes=audio_signal:1x128x57,length:1,cache_last_channel:1x24x256x1024,cache_last_time:1x24x1024x4,cache_last_channel_len:1
```

### TRT closed-loop parity (cache-aware tolerances + valid-region masking)
```bash
python tools/tensorrt/trt_streaming_parity.py \
  --engine /home/emmy/git/trt-asr-engine/models/parakeet-tdt-0.6b-v3/engines_20260108_cache3_fp32_t57/encoder_streaming.engine \
  --ref artifacts/reference/stream_ref_cache3_50.jsonl \
  --mode closed_loop \
  --valid-out-len 3 \
  --cache-size 256 \
  --time-ctx 4 \
  --atol 1e-3 \
  --rtol 1e-4 \
  --cache-atol 1e-1 \
  --summary-json artifacts/parity/trt_streaming_parity_cache3_fp32_t57_50.json
```

### Results
- PASS `45/50`.
- encoder_output max_abs `6.388e-04` (within TRT p100 tolerance `1e-3`).
- cache_last_channel_out compared on valid region only; max_abs `5.225e-03` (passes with cache_channel_atol=1e-2 default).
- cache_last_time_out max_abs `3.614e-01` (5 chunks exceed cache_atol=0.1).
- cache_last_channel_len_out matches reference for all chunks.

### Interpretation
- TRT closed-loop parity is now dominated by cache_last_time_out outliers (5 chunks). Next decision: either relax TRT cache_time tolerance or rebuild with TF32 disabled to see if the outliers shrink.
