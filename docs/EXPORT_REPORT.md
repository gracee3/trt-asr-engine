# Export Report (Draft)

Status: NOT RUN in this pass. Fill in after deterministic export (Phase C).

## Environment
- torch version:
- nemo version:
- onnx version:
- exporter flags:

## Commands
```bash
# Offline graphs
python -u tools/export_onnx/export.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out out/onnx/parakeet-tdt-0.6b-v3/base \
  --component all \
  --device cpu \
  --smoke-test-ort

# Streaming encoder
python -u tools/export_onnx/export.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out out/onnx/parakeet-tdt-0.6b-v3/streaming \
  --component encoder_streaming \
  --streaming-cache-size 256 \
  --device cpu
```

## Artifacts
- encoder.onnx:
- predictor.onnx:
- joint.onnx:
- encoder_streaming.onnx:
- model_meta.json:
- external data shards (*.onnx.data or weight files):

## Hashes
- encoder.onnx:
- predictor.onnx:
- joint.onnx:
- encoder_streaming.onnx:
- model_meta.json:

## Smoke tests
- ONNX checker:
- ORT one-pass smoke test:

## Notes
- Dynamic axes:
- External data policy:
