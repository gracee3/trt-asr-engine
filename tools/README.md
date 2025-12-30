# trt-asr-engine tools

This directory contains the pipeline for converting a NeMo Parakeet-TDT model into TensorRT engines and verifying the results.

## Subdirectories

- **`verify_nemo/`**: Python scripts to run the model in its native NeMo/PyTorch environment. Used to establish a "golden transcript" for comparison.
- **`export_onnx/`**: Tools to split the `.nemo` model into ONNX components (Encoder, Predictor, Joint).
- **`build_trt/`**: Scripts and documentation for building TensorRT engines from ONNX artifacts.

## Recommended Workflow

1.  **Verify**: Run `verify_nemo/verify.py` on a sample WAV to ensure the local `.nemo` file is working and to get a reference transcript.
2.  **Export**: Run `export_onnx/export.py` to generate the three ONNX components in `export_onnx/out/`.
3.  **Build**: Use the tools in `build_trt/` to generate `.engine` files for the target hardware.
