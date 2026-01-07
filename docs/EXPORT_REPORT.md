# Export Report

Status: COMPLETED (CPU exports with legacy torch.onnx.export).

## Environment
- torch version: 2.9.1+cu128
- nemo version: 2.6.0
- onnx version: 1.16.2
- exporter flags: `torch.onnx.export(dynamo=False, fallback=False)` (legacy exporter)

## Commands
```bash
# Offline graphs
python -u tools/export_onnx/export.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out tools/export_onnx/out \
  --component all \
  --device cpu \
  --smoke-test-ort

# Streaming encoder
python -u tools/export_onnx/export.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out tools/export_onnx/out \
  --component encoder_streaming \
  --streaming-cache-size 256 \
  --device cpu
```

## Artifacts
- encoder.onnx: `tools/export_onnx/out/encoder.onnx`
- predictor.onnx: `tools/export_onnx/out/predictor.onnx`
- joint.onnx: `tools/export_onnx/out/joint.onnx` (raw logits; no LogSoftmax)
- encoder_streaming.onnx: `tools/export_onnx/out/encoder_streaming.onnx` (batch-first caches)
- model_meta.json: `tools/export_onnx/out/model_meta.json`
- external data shards (*.onnx.data or weight files):

## Hashes
- encoder.onnx: not captured yet
- predictor.onnx: not captured yet
- joint.onnx: not captured yet
- encoder_streaming.onnx: not captured yet
- model_meta.json: not captured yet

## Smoke tests
- ONNX checker: PASS (encoder/predictor/joint/encoder_streaming)
- ORT one-pass smoke test: PASS (predictor + joint, CPUExecutionProvider)
- Joint graph check: PASS (`python tools/inspect_onnx/check_joint_output.py`)

## Notes
- Dynamic axes: enabled for batch/time dims; cache tensors batch axis dynamic.
- External data policy: none observed in these exports.
- Streaming encoder config after setup: `cache_drop_size=71`, `shift_size=[9,16]`, `valid_out_len=2`.
