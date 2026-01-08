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
  --streaming-chunk-size 48 \
  --streaming-cache-drop-size 3 \
  --streaming-dummy-len 48 \
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
- encoder.onnx: `0a37e8fbba247f469c39c582c8c0e2813b4d60172356c6edeff4c45275105110`
- predictor.onnx: `e5b8f22214be6d6884c56140de37e17a5069da4de77ce92fafd70a619e914eaa`
- joint.onnx: `7b6f2326ac958bc6ddfa82c07e6d1222ca583f55ccb674474b03a38089ba4071`
- encoder_streaming.onnx: `7bdc37e098e9df62b888bd256a0de16d4901ac0589cdfc4895f8d1abaec2a943` (streaming cache_drop_size=3 override)
- model_meta.json: `d3421f784ac920f2c6452916f7635ea2f0eb196471de31be1475712823537806`

## Smoke tests
- ONNX checker: PASS (encoder/predictor/joint/encoder_streaming)
- ORT one-pass smoke test: PASS (predictor + joint, CPUExecutionProvider)
- Joint graph check: PASS (`python tools/inspect_onnx/check_joint_output.py`)

## Notes
- Dynamic axes: enabled for batch/time dims; cache tensors batch axis dynamic.
- External data policy: none observed in these exports.
- Streaming encoder config after setup: `cache_drop_size=3`, `shift_size=[17,24]`, `valid_out_len=3`, `chunk_size=[41,48]`.
