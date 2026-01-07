# ORT Validation Report

Status: PARTIAL (streaming encoder parity; cache_last_time_out mismatch outstanding).

## Environment
- onnxruntime version: 1.23.2
- device/provider: CPUExecutionProvider
- model export hash: not captured yet

## Component parity
- Encoder (offline): not run
- Encoder (streaming): PARTIAL
  - Functional (4 chunks): FAIL (cache_last_time_out chunk 0 @ atol=1e-4)
  - Closed-loop (4 chunks): FAIL (cache_last_time_out all chunks @ atol=1e-4)
- Predictor: PASS (logits parity, 1 step)
- Joint: PASS (logits parity, 1 step; argmax token/duration match)
  - PyTorch joint run with fuse_loss_wer disabled and log_softmax forced off to match export

## Closed-loop parity
- Encoder cache closed-loop: FAIL (`tools/export_onnx/out/encoder_streaming.onnx` vs `artifacts/reference/pytorch_reference_4.jsonl`) due to cache_last_time_out
- Predictor state closed-loop: not run
- Decode closed-loop (token IDs): not run

## Metrics
- max abs error: encoder_output ≤ 5.1e-7; cache_last_time_out ≤ 3.7e-4 (4 chunks, CPU)
- mean abs error: cache_last_time_out ≈ 1.0e-5 (chunk 0 functional)
- predictor g/h_out/c_out: max_abs ≤ 4.83e-6; max_rel ≤ 1.44
- joint_output: max_abs ≤ 8.55e-4; max_rel ≤ 1.21e-6
- p95/p99 abs error: not computed
- drift slope over chunks: not computed

## Pass/fail gates
- Contract assertions: streaming cache_len monotonic (0,2,4,6) and shapes aligned
- Decode parity: not run

## Artifacts
- parity summaries (json): `artifacts/parity/ort_streaming_functional_4.json`, `artifacts/parity/ort_streaming_closedloop_4.json`, `artifacts/parity/ort_predictor_joint_parity.json`
- drift plots: not generated
