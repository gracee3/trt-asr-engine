# trt-asr-engine tools

This directory contains the pipeline for converting a NeMo Parakeet-TDT model into TensorRT engines, verifying the results, and debugging integration issues.

## Subdirectories

- **`verify_nemo/`**: Python scripts to run the model in its native NeMo/PyTorch environment. Used to establish a "golden transcript" for comparison.
- **`export_onnx/`**: Tools to split the `.nemo` model into ONNX components (Encoder, Predictor, Joint).
- **`build_trt/`**: Scripts and documentation for building TensorRT engines from ONNX artifacts.

## Standalone Scripts

- **`analyze_tap.py`**: Audio/feature tap analysis tool for debugging pipeline issues.

## Recommended Workflow

1.  **Verify**: Run `verify_nemo/verify.py` on a sample WAV to ensure the local `.nemo` file is working and to get a reference transcript.
2.  **Export**: Run `export_onnx/export.py` to generate the three ONNX components in `export_onnx/out/`.
3.  **Build**: Use the tools in `build_trt/` to generate `.engine` files for the target hardware.

---

## analyze_tap.py

Analyzes audio and feature tap dumps produced by the debugging infrastructure. See [docs/debugging.md](../docs/debugging.md) for full documentation.

### Installation

```bash
pip install numpy matplotlib scipy  # matplotlib/scipy optional for plots
```

### Usage

```bash
# Analyze single audio tap
python tools/analyze_tap.py tap_post_dsp.raw

# Analyze feature tap (mel-spectrogram visualization)
python tools/analyze_tap.py tap_features.raw --features

# Compare multiple taps (detect energy drops between pipeline stages)
python tools/analyze_tap.py tap_capture.raw tap_post_dsp.raw --compare
```

### Output

- **Statistics**: Peak, RMS, DC offset, NaN/Inf counts, clipping detection
- **Plots**: Waveform + spectrogram PNG (or mel-spectrogram for features)
- **Warnings**: Silent audio, phase cancellation, DC drift, energy drops

### Options

| Option | Description |
|--------|-------------|
| `--json PATH` | Specify JSON sidecar (auto-detected if not given) |
| `-o, --output PATH` | Output plot filename |
| `-f, --features` | Treat input as mel-feature dump (128 bins) |
| `--no-plot` | Skip plot generation |
| `-c, --compare` | Compare multiple taps side-by-side |
