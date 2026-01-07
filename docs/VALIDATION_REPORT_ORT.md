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
- Predictor: not run
- Joint: not run

## Closed-loop parity
- Encoder cache closed-loop: FAIL (`tools/export_onnx/out/encoder_streaming.onnx` vs `artifacts/reference/pytorch_reference_4.jsonl`) due to cache_last_time_out
- Predictor state closed-loop: not run
- Decode closed-loop (token IDs): not run

## Metrics
- max abs error: encoder_output ≤ 5.1e-7; cache_last_time_out ≤ 3.7e-4 (4 chunks, CPU)
- mean abs error: cache_last_time_out ≈ 1.0e-5 (chunk 0 functional)
- p95/p99 abs error: not computed
- drift slope over chunks: not computed

## Pass/fail gates
- Contract assertions: streaming cache_len monotonic (0,2,4,6) and shapes aligned
- Decode parity: not run

## Artifacts
- parity summaries (json): `artifacts/parity/ort_streaming_functional_4.json`, `artifacts/parity/ort_streaming_closedloop_4.json`
- drift plots: not generated
