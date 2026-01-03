# Agent Setup Guide: TRT Streaming Encoder Integration

**Purpose:** Enable AI agents to repeat TRT integration and validation tasks for the parakeet-tdt-0.6b-v3 streaming encoder.

**Last Updated:** 2026-01-03

---

## Quick Start Checklist

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Verify TensorRT is available
trtexec --version

# 3. Verify Python packages
python3 -c "import tensorrt as trt; print(f'TRT: {trt.__version__}')"
python3 -c "from cuda.bindings import runtime as cudart; print('CUDA bindings OK')"

# 4. Run a quick parity test
python3 tools/tensorrt/trt_streaming_parity.py \
  --engine out/trt_engines/encoder_streaming_fp32.plan \
  --ref artifacts/reference/pytorch_reference_50.jsonl \
  --mode functional \
  --max-chunks 5
```

---

## Environment Requirements

### System Requirements

| Component | Version | Location |
|-----------|---------|----------|
| Python | 3.10.19 | `~/.pyenv/shims/python3` |
| TensorRT | 10.14.1 | `/usr/src/tensorrt/bin/trtexec` |
| CUDA | 12.x | System-installed |
| GPU | NVIDIA with Tensor Cores | Required for TRT inference |

### Python Virtual Environment

**Location:** `.venv/`

**Activation:**
```bash
cd /home/emmy/git/trt-asr-engine
source .venv/bin/activate
```

### Required Pip Packages

```
tensorrt_cu12==10.14.1.48.post1
tensorrt_cu12_bindings==10.14.1.48.post1
tensorrt_cu12_libs==10.14.1.48.post1
cuda-python==13.1.1
numpy==1.26.4
onnx==1.16.2
onnxruntime-gpu==1.23.2
torch==2.9.1
nemo-toolkit==2.6.0
```

**Install from scratch (if needed):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tensorrt-cu12 cuda-python numpy onnx onnxruntime-gpu torch nemo-toolkit
```

---

## Directory Structure

```
/home/emmy/git/trt-asr-engine/
├── .venv/                          # Python virtual environment
├── models/
│   └── parakeet-tdt-0.6b-v3/
│       └── parakeet-tdt-0.6b-v3.nemo   # Source NeMo model (2.5GB)
├── out/
│   ├── encoder_streaming.onnx      # ONNX model (with external weights)
│   ├── *.weight                    # External ONNX weights (~2.3GB total)
│   └── trt_engines/
│       ├── encoder_streaming_fp32.plan  # FP32 TRT engine (2.4GB)
│       └── encoder_streaming_fp16.plan  # FP16 TRT engine (1.2GB)
├── contracts/
│   └── encoder_streaming.contract.json  # I/O binding specification
├── artifacts/
│   ├── reference/
│   │   ├── pytorch_reference_50.jsonl   # 50-chunk reference data (3.2GB)
│   │   └── pytorch_reference_300.jsonl  # 300-chunk reference data (20GB)
│   ├── parity/                          # ORT/TRT parity summaries
│   ├── stability/                       # Error trend plots
│   └── diagnostics/                     # Cache mismatch diagnostics
├── tools/
│   ├── tensorrt/
│   │   ├── trt_streaming_parity.py      # TRT parity test harness
│   │   └── plot_stability.py            # Error trend analysis
│   ├── onnxruntime/
│   │   ├── onnx_streaming_parity.py     # ORT parity test harness
│   │   └── diagnose_cache_time_mismatch.py
│   └── verify_nemo/
│       └── streaming_encoder_reference.py  # PyTorch reference generator
└── [documentation files]
```

---

## Key Files Reference

### Configuration & Contracts

| File | Purpose |
|------|---------|
| `contracts/encoder_streaming.contract.json` | **CRITICAL** - I/O shapes, optimization profiles, runtime assertions |

### Models & Engines

| File | Size | Description |
|------|------|-------------|
| `models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo` | 2.5GB | Source NeMo checkpoint |
| `out/encoder_streaming.onnx` | 2.5MB | ONNX model (graph only) |
| `out/*.weight` | ~2.3GB | External ONNX weights |
| `out/trt_engines/encoder_streaming_fp32.plan` | 2.4GB | TRT FP32 engine |
| `out/trt_engines/encoder_streaming_fp16.plan` | 1.2GB | TRT FP16 engine |

### Reference Data (Gitignored - Generate Locally)

| File | Size | Chunks | Command to Generate |
|------|------|--------|---------------------|
| `artifacts/reference/pytorch_reference_50.jsonl` | 3.2GB | 50 | See "Generate Reference Data" section |
| `artifacts/reference/pytorch_reference_300.jsonl` | 20GB | 300 | See "Generate Reference Data" section |

### Test Tools

| File | Purpose |
|------|---------|
| `tools/tensorrt/trt_streaming_parity.py` | TRT parity testing with contract enforcement |
| `tools/tensorrt/plot_stability.py` | Error accumulation analysis |
| `tools/verify_nemo/streaming_encoder_reference.py` | Generate PyTorch ground truth |

### Documentation

| File | Purpose |
|------|---------|
| `TRT_INTEGRATION_CHECKLIST.md` | Step-by-step integration guide with results |
| `TRT_INTEGRATION_CLEARANCE.md` | Formal clearance and requirements |
| `ONNX_ORT_PARITY_README.md` | Overview and quick start |
| `CACHE_TIME_ROOT_CAUSE_ANALYSIS.md` | Known issue documentation |
| `ONNX_PARITY_RESULTS.md` | ORT baseline parity results |

---

## Generate Reference Data

If reference JSONL files are missing, generate them:

**50-chunk reference (required for quick tests):**
```bash
source .venv/bin/activate
python3 tools/verify_nemo/streaming_encoder_reference.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --device cuda \
  --cache-size 256 \
  --chunk-len 592 \
  --num-chunks 50 \
  --seed 42 \
  --skip-setup-streaming-params \
  --jsonl-out artifacts/reference/pytorch_reference_50.jsonl
```

**300-chunk reference (required for stability tests):**
```bash
python3 tools/verify_nemo/streaming_encoder_reference.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --device cuda \
  --cache-size 256 \
  --chunk-len 592 \
  --num-chunks 300 \
  --seed 42 \
  --skip-setup-streaming-params \
  --jsonl-out artifacts/reference/pytorch_reference_300.jsonl
```

---

## Build TRT Engines

If TRT engines are missing, rebuild them:

**FP32 Engine:**
```bash
trtexec \
  --onnx=out/encoder_streaming.onnx \
  --saveEngine=out/trt_engines/encoder_streaming_fp32.plan \
  --minShapes=audio_signal:1x128x584,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
  --optShapes=audio_signal:1x128x592,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
  --maxShapes=audio_signal:1x128x592,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
  --builderOptimizationLevel=5 \
  --memPoolSize=workspace:4G
```

**FP16 Engine:**
```bash
trtexec \
  --onnx=out/encoder_streaming.onnx \
  --saveEngine=out/trt_engines/encoder_streaming_fp16.plan \
  --minShapes=audio_signal:1x128x584,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
  --optShapes=audio_signal:1x128x592,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
  --maxShapes=audio_signal:1x128x592,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
  --fp16 \
  --builderOptimizationLevel=5 \
  --memPoolSize=workspace:4G
```

**Expected build times:** FP32 ~153s, FP16 ~180s

---

## Run Parity Tests

### Full Validation Suite

```bash
source .venv/bin/activate

# 1. FP32 50-chunk functional parity
python3 tools/tensorrt/trt_streaming_parity.py \
  --engine out/trt_engines/encoder_streaming_fp32.plan \
  --ref artifacts/reference/pytorch_reference_50.jsonl \
  --mode functional \
  --summary-json artifacts/parity/trt_parity_50chunks_functional.json

# 2. FP32 50-chunk closed-loop parity
python3 tools/tensorrt/trt_streaming_parity.py \
  --engine out/trt_engines/encoder_streaming_fp32.plan \
  --ref artifacts/reference/pytorch_reference_50.jsonl \
  --mode closed_loop \
  --summary-json artifacts/parity/trt_parity_50chunks_closedloop.json

# 3. FP32 300-chunk stability test
python3 tools/tensorrt/trt_streaming_parity.py \
  --engine out/trt_engines/encoder_streaming_fp32.plan \
  --ref artifacts/reference/pytorch_reference_300.jsonl \
  --mode closed_loop \
  --summary-json artifacts/parity/trt_parity_300chunks_closedloop.json

# 4. Generate stability plot
python3 tools/tensorrt/plot_stability.py \
  --summary-json artifacts/parity/trt_parity_300chunks_closedloop.json \
  --output-png artifacts/stability/trt_stability_300chunks.png
```

### FP16 Validation

```bash
# FP16 50-chunk functional
python3 tools/tensorrt/trt_streaming_parity.py \
  --engine out/trt_engines/encoder_streaming_fp16.plan \
  --ref artifacts/reference/pytorch_reference_50.jsonl \
  --mode functional \
  --summary-json artifacts/parity/trt_parity_50chunks_functional_fp16.json

# FP16 300-chunk stability
python3 tools/tensorrt/trt_streaming_parity.py \
  --engine out/trt_engines/encoder_streaming_fp16.plan \
  --ref artifacts/reference/pytorch_reference_300.jsonl \
  --mode closed_loop \
  --summary-json artifacts/parity/trt_parity_300chunks_closedloop_fp16.json

# FP16 stability plot
python3 tools/tensorrt/plot_stability.py \
  --summary-json artifacts/parity/trt_parity_300chunks_closedloop_fp16.json \
  --output-png artifacts/stability/trt_stability_300chunks_fp16.png
```

---

## Expected Results (Baseline)

### FP32 Results

| Test | Pass Rate | P95 Error | P99 Error | Trend Slope |
|------|-----------|-----------|-----------|-------------|
| 50-chunk Functional | 90% | 4.88e-4 | 9.70e-4 | — |
| 50-chunk Closed-loop | 88% | 5.05e-4 | 9.20e-4 | — |
| 300-chunk Stability | 83% | 6.58e-4 | 1.69e-3 | -3e-7 (~0) |

### FP16 Results

| Test | Pass Rate | P95 Error | P99 Error | Trend Slope |
|------|-----------|-----------|-----------|-------------|
| 50-chunk Functional | 6% | 1.09e-3 | 1.77e-3 | — |
| 300-chunk Stability | 0.67% | 1.81e-3 | 3.93e-3 | -9e-7 (~0) |

**Note:** FP16 has lower pass rate due to strict tolerance (5e-4), but all contract assertions pass 100%.

### Contract Assertions (Must be 100%)

| Assertion | Expected |
|-----------|----------|
| `encoded_lengths == 1` | 100% pass |
| `cache_last_channel_len_out == 0` | 100% pass |
| `encoder_output.shape[-1] == 1` | 100% pass |

---

## Streaming Contract Summary

**Critical Understanding:** The model operates in **chunk-isolated mode**:
- `cache_last_channel_len == 0` for ALL chunks (input and output)
- Cache is used **intra-chunk** only (within single forward pass)
- Cache is **NOT carried inter-chunk** (reset between chunks)

**Mandatory Runtime Assertions:**
```python
# MUST enforce at every inference step
assert np.all(outputs["encoded_lengths"] == 1)
assert outputs["encoder_output"].shape[-1] == 1
assert np.all(outputs["cache_last_channel_len_out"] == 0)
```

**Known Issue (Non-Blocking):**
- `cache_last_time_out` has 0.01-0.1 abs error vs PyTorch
- Non-propagating due to `cache_len=0` isolation
- Use relaxed tolerance: `--cache-atol 0.1`

---

## Troubleshooting

### Import Error: `from cuda.bindings import runtime as cudart`

**Fix:** Install cuda-python
```bash
pip install cuda-python
```

### TRT Engine Build Fails with OOM

**Fix:** Reduce workspace or use streaming build
```bash
trtexec ... --memPoolSize=workspace:2G
```

### Reference JSONL Missing

**Fix:** Generate with streaming_encoder_reference.py (see above)

### Low Pass Rate on FP16

**Expected:** FP16 has 2-3x higher numerical error than FP32. Contract assertions should still be 100%.

### Error Accumulation Detected (slope > 1e-5)

**Action:** This is a CRITICAL failure. Re-validate:
1. Check if `cache_last_channel_len_out == 0` for all chunks
2. Compare to ORT baseline
3. Inspect ONNX graph for changes

---

## Verification Checklist for Agents

When running TRT integration validation, verify:

- [ ] Virtual environment activated: `source .venv/bin/activate`
- [ ] TensorRT available: `trtexec --version`
- [ ] Python packages installed: `python3 -c "import tensorrt"`
- [ ] ONNX model exists: `out/encoder_streaming.onnx`
- [ ] TRT engine exists: `out/trt_engines/encoder_streaming_fp32.plan`
- [ ] Reference data exists: `artifacts/reference/pytorch_reference_50.jsonl`
- [ ] Contract file exists: `contracts/encoder_streaming.contract.json`
- [ ] 50-chunk functional parity: Pass rate > 80%
- [ ] 300-chunk stability: Trend slope ≈ 0 (< 1e-5)
- [ ] All contract assertions: 100% pass

---

## Performance Targets

| Metric | FP32 | FP16 | Target |
|--------|------|------|--------|
| GPU Latency | 18.6ms | 12.1ms | < 21ms |
| Engine Size | 2.4GB | 1.2GB | — |
| Throughput | 54 qps | 82 qps | — |

---

## Related Documentation

- [TRT_INTEGRATION_CHECKLIST.md](TRT_INTEGRATION_CHECKLIST.md) - Detailed integration steps
- [TRT_INTEGRATION_CLEARANCE.md](TRT_INTEGRATION_CLEARANCE.md) - Requirements and sign-off
- [ONNX_ORT_PARITY_README.md](ONNX_ORT_PARITY_README.md) - Overview and quick start
- [contracts/encoder_streaming.contract.json](contracts/encoder_streaming.contract.json) - Binding specification
- [CACHE_TIME_ROOT_CAUSE_ANALYSIS.md](CACHE_TIME_ROOT_CAUSE_ANALYSIS.md) - Known issue deep-dive

---

## Next Phase: Magnolia Integration

Once TRT validation is complete, proceed to Magnolia integration:

**Handoff document:** [MAGNOLIA_INTEGRATION_HANDOFF.md](MAGNOLIA_INTEGRATION_HANDOFF.md)

This covers:
- Live captions pipeline (Audio → DSP → parakeet_stt → Transcription)
- Partials vs finals (stable prefix + revision window)
- Low-latency engineering constraints
- Multi-engine roadmap
- LLM integration rules
