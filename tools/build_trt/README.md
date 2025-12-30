# TensorRT Engine Builder

This folder will contain scripts to convert the ONNX components into optimized TensorRT engines.

## Prerequisites
- NVIDIA TensorRT (recommended 10.x+)
- `trtexec` available in PATH.

## Next Steps
Once components are exported to `tools/export_onnx/out/`, we will use `trtexec` or a Python script to build:
- `encoder.engine`
- `predictor.engine`
- `joint.engine`

Target: FP16 precision with dynamic shapes for the time dimension.
