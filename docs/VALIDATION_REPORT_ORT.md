# ORT Validation Report

Status: PARTIAL (streaming encoder closed-loop parity only).

## Environment
- onnxruntime version: 1.23.2
- device/provider: CPUExecutionProvider
- model export hash: not captured yet

## Component parity
- Encoder (offline): not run
- Encoder (streaming): PASS (closed-loop, 4 chunks)
- Predictor: not run
- Joint: not run

## Closed-loop parity
- Encoder cache closed-loop: PASS (`tools/export_onnx/out/encoder_streaming.onnx` vs `artifacts/reference/pytorch_reference_4.jsonl`)
- Predictor state closed-loop: not run
- Decode closed-loop (token IDs): not run

## Metrics
- max abs error: encoder_output ≤ 2.7e-7; cache_last_time_out ≤ 7.6e-5 (4 chunks, CPU)
- p95/p99 abs error: not computed
- drift slope over chunks: not computed

## Pass/fail gates
- Contract assertions: streaming cache_len monotonic (0,2,4,6) and shapes aligned
- Decode parity: not run

## Artifacts
- parity summaries (json): not saved
- drift plots: not generated
