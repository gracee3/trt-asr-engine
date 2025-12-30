# TensorRT Engine Builder (`trtexec`-driven)

This tool builds FP16 TensorRT engines for Parakeet RNNT-TDT components using **`trtexec`** (not the TensorRT Python API) for speed of implementation and reproducibility across dev machines.

## Prerequisites

- NVIDIA TensorRT installed (recommended 10.x+)
- `trtexec` available in `PATH`
- Python 3.9+
- ONNX Python package available (`pip install onnx`) (the exporter already depends on it)

## Minimal developer command

From repo root:

```bash
python tools/build_trt/build_trt.py \
  --meta tools/export_onnx/out/model_meta.json \
  --outdir models/parakeet-tdt-0.6b-v3 \
  --fp16
```

This will:

- Read the meta file as the source of truth
- Locate `encoder.onnx`, `predictor.onnx`, `joint.onnx` **in the same directory as the meta file**
- Stage ONNX(+external-data sidecars) into a clean temp directory (to guarantee `.onnx.data` discovery)
- Build TensorRT engines with explicit batch + dynamic shapes via optimization profiles
- Write engines to:
  - `models/parakeet-tdt-0.6b-v3/encoder.engine`
  - `models/parakeet-tdt-0.6b-v3/predictor.engine`
  - `models/parakeet-tdt-0.6b-v3/joint.engine`
- Emit a build report (timings, TRT version, GPU name, profiles used, engine sizes)

## Outputs

All outputs are written under `--outdir`:

- `*.engine`: built engines
- `timing.cache`: persisted TensorRT timing cache
- `build_logs/*.log`: raw `trtexec` stdout/stderr logs per component
- `build_report.json`: machine-readable build report

## Dynamic shape profiles (defaults)

- **Encoder**: `audio_signal [B,128,T]` with \(T\) min/opt/max = 16/64/256 and `B=1`
- **Predictor**: `y [B,U]` with \(U\) min/opt/max = 1/1/8 and `B=1`, with `h/c [L,B,H]` fixed from the ONNX signature
- **Joint**: `T` min/opt/max = 16/64/256 and `U` min/opt/max = 1/1/8 with `B=1`

You can override profile maxima/minima with CLI flags (see `--help`).

## Scripts

- `tools/build_trt/scripts/build_all.sh`: one-button wrapper
- `tools/build_trt/scripts/inspect_engine.py`: quick engine inspection/smoke-run helper
