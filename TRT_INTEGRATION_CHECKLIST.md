# TensorRT Integration Checklist

**Model:** parakeet-tdt-0.6b-v3 streaming encoder
**ONNX:** `encoder_streaming.onnx`
**Status:** ✅ Cleared for integration (see [TRT_INTEGRATION_CLEARANCE.md](TRT_INTEGRATION_CLEARANCE.md))

---

## Phase 1: Pre-Build (Review & Planning)

### 1.1 Review Binding Contract
- [ ] Read [`contracts/encoder_streaming.contract.json`](contracts/encoder_streaming.contract.json)
- [ ] Understand I/O shapes and dynamic dimensions
- [ ] Review optimization profiles (T=592, T=584, unified)
- [ ] Understand cache isolation contract (`cache_len=0` always)

### 1.2 Review Parity Test Results
- [ ] Read [`ONNX_PARITY_RESULTS.md`](ONNX_PARITY_RESULTS.md)
- [ ] Read [`TRT_INTEGRATION_CLEARANCE.md`](TRT_INTEGRATION_CLEARANCE.md)
- [ ] Understand known cache_last_time issue (non-blocking)
- [ ] Note tolerance guidance (encoder_output < 5e-4 @ p95)

### 1.3 Set Up Reference Data
- [ ] Verify `pytorch_reference_50.jsonl` exists (3.2GB)
- [ ] Generate `pytorch_reference_300.jsonl` for long-run stability test:
  ```bash
  python3 tools/verify_nemo/streaming_encoder_reference.py \
    --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
    --device cuda \
    --cache-size 256 \
    --chunk-len 592 \
    --num-chunks 300 \
    --seed 42 \
    --skip-setup-streaming-params \
    --jsonl-out pytorch_reference_300.jsonl
  ```

---

## Phase 2: Engine Build (FP32)

### 2.1 Create Optimization Profiles
- [ ] **Profile A (First Chunk):**
  - `audio_signal.T`: min=592, opt=592, max=592
  - `B`: min=1, opt=1, max=1
- [ ] **Profile B (Subsequent Chunks):**
  - `audio_signal.T`: min=584, opt=584, max=584
  - `B`: min=1, opt=1, max=1
- [ ] (Optional) **Profile C (Unified):**
  - `audio_signal.T`: min=584, opt=592, max=592
  - `B`: min=1, opt=1, max=8

### 2.2 Build TensorRT Engine
- [ ] Use `trtexec` or Python API to build engine:
  ```bash
  trtexec --onnx=encoder_streaming.onnx \
          --saveEngine=encoder_streaming.plan \
          --minShapes=audio_signal:1x128x584,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
          --optShapes=audio_signal:1x128x592,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
          --maxShapes=audio_signal:1x128x592,length:1,cache_last_channel:24x1x256x1024,cache_last_time:24x1x1024x4,cache_last_channel_len:1 \
          --fp16  # Optional, test FP32 first
  ```
- [ ] Verify engine builds without errors
- [ ] Check engine size and layer count

### 2.3 Create Runtime Wrapper
- [ ] Implement TRT inference wrapper with **mandatory assertions**:

```python
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class StreamingEncoderTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        self._allocate_buffers()

    def infer(self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        # Copy inputs to device
        self.inputs['audio_signal'].host = audio_signal
        self.inputs['length'].host = length
        self.inputs['cache_last_channel'].host = cache_last_channel
        self.inputs['cache_last_time'].host = cache_last_time
        self.inputs['cache_last_channel_len'].host = cache_last_channel_len

        # Transfer to device
        for inp in self.inputs.values():
            cuda.memcpy_htod(inp.device, inp.host)

        # Execute
        self.context.execute_v2(bindings=self.bindings)

        # Transfer outputs
        for out in self.outputs.values():
            cuda.memcpy_dtoh(out.host, out.device)

        outputs = {name: buf.host for name, buf in self.outputs.items()}

        # MANDATORY ASSERTIONS (Contract Enforcement)
        self._validate_streaming_contract(outputs)

        return outputs

    def _validate_streaming_contract(self, outputs):
        """Hard fail on contract violations."""
        # Assertion 1: Encoded lengths must be 1 (valid_out_len=1)
        assert np.all(outputs['encoded_lengths'] == 1), \
            f"STREAMING CONTRACT VIOLATION: encoded_lengths != 1, got {outputs['encoded_lengths']}"

        # Assertion 2: Encoder output time dimension must be 1
        assert outputs['encoder_output'].shape[-1] == 1, \
            f"STREAMING CONTRACT VIOLATION: encoder_output time != 1, shape={outputs['encoder_output'].shape}"

        # Assertion 3: Cache length must be 0 (chunk-isolated mode)
        assert np.all(outputs['cache_last_channel_len_out'] == 0), \
            f"STREAMING CONTRACT VIOLATION: cache_len != 0, got {outputs['cache_last_channel_len_out']}"

    def _allocate_buffers(self):
        # Implementation details for buffer allocation
        pass
```

---

## Phase 3: Functional Parity Testing (50 chunks)

### 3.1 Adapt ORT Parity Harness for TRT
- [ ] Copy [`tools/onnxruntime/onnx_streaming_parity.py`](tools/onnxruntime/onnx_streaming_parity.py) → `tools/tensorrt/trt_streaming_parity.py`
- [ ] Replace ORT session with TRT engine/wrapper
- [ ] Keep JSONL decoding logic identical
- [ ] Keep comparison logic identical
- [ ] Add TRT-specific profiling (latency per chunk)

### 3.2 Run Functional Mode (Reference Caches In)
- [ ] Run functional parity test:
  ```bash
  python3 tools/tensorrt/trt_streaming_parity.py \
    --engine encoder_streaming.plan \
    --ref pytorch_reference_50.jsonl \
    --mode functional \
    --summary-json trt_parity_50chunks_functional.json
  ```
- [ ] Review results:
  - **MUST PASS:** All contract assertions (cache_len=0, encoded_len=1, time_dim=1)
  - **EXPECT:** encoder_output errors similar to ORT (6e-5 to 7e-4)
  - **ACCEPTABLE:** cache_last_time_out errors up to 0.1 (non-blocking)

### 3.3 Validate Results
- [ ] Check pass rate for encoder_output (target: 80%+ chunks within 5e-4)
- [ ] Verify no contract assertion failures
- [ ] Compare to ORT baseline (should be similar or better)
- [ ] If failures: debug with `--dump-dir` to inspect failing chunks

---

## Phase 4: Closed-Loop Stability Testing (300 chunks)

### 4.1 Run Closed-Loop Mode (Cache Feedback)
- [ ] Run 300-chunk closed-loop test:
  ```bash
  python3 tools/tensorrt/trt_streaming_parity.py \
    --engine encoder_streaming.plan \
    --ref pytorch_reference_300.jsonl \
    --mode closed_loop \
    --summary-json trt_parity_300chunks_closedloop.json
  ```

### 4.2 Analyze Stability Metrics
- [ ] Extract per-chunk encoder_output max_abs errors
- [ ] Plot error vs chunk_index:
  ```python
  import json
  import matplotlib.pyplot as plt

  with open('trt_parity_300chunks_closedloop.json') as f:
      data = json.load(f)

  errors = [chunk['encoder_output_max_abs'] for chunk in data['failures']]
  plt.plot(errors)
  plt.xlabel('Chunk Index')
  plt.ylabel('encoder_output max_abs error')
  plt.title('TRT Closed-Loop Stability (300 chunks)')
  plt.axhline(y=5e-4, color='r', linestyle='--', label='Target Tolerance')
  plt.legend()
  plt.savefig('trt_stability_300chunks.png')
  ```

### 4.3 Acceptance Criteria
- [ ] **No monotonic error growth** (slope ≈ 0 in error plot)
- [ ] **Errors remain bounded** (max < 1e-3 across all 300 chunks)
- [ ] **No contract assertion failures**
- [ ] **Mean error stable** (< 2e-4 across all chunks)

---

## Phase 5: Performance Validation

### 5.1 Latency Benchmarking
- [ ] Measure per-chunk latency (warm-up first):
  ```bash
  # Use trtexec or custom harness
  # Target: < 21ms per chunk (match PyTorch baseline)
  ```
- [ ] Profile breakdown:
  - [ ] Memory transfer (H2D + D2H)
  - [ ] Compute time
  - [ ] Cache handling overhead

### 5.2 Throughput Testing
- [ ] Test batched inference (if using batch profile):
  - [ ] B=1, B=2, B=4, B=8
  - [ ] Measure latency vs throughput tradeoff

### 5.3 Memory Footprint
- [ ] Measure engine size (target: < 2GB for FP32)
- [ ] Measure runtime memory (peak GPU usage)
- [ ] Compare to PyTorch baseline

---

## Phase 6: Integration & Monitoring

### 6.1 Runtime Integration
- [ ] Integrate TRT wrapper into production pipeline
- [ ] Add logging for contract assertions
- [ ] Set up telemetry for error metrics

### 6.2 CI/CD Integration
- [ ] Add TRT parity test to CI pipeline
- [ ] Set up regression detection:
  - [ ] Fail if cache_len != 0 detected
  - [ ] Fail if encoder_output errors exceed 1e-3
  - [ ] Warn if errors exceed 5e-4

### 6.3 Documentation
- [ ] Document TRT engine build process
- [ ] Document runtime assertions and error handling
- [ ] Document performance characteristics
- [ ] Update deployment guide

---

## Phase 7: Optional Enhancements

### 7.1 FP16/INT8 Quantization
- [ ] Build FP16 engine and re-run parity (expect errors to increase)
- [ ] Evaluate INT8 quantization (requires calibration data)
- [ ] Balance accuracy vs performance

### 7.2 Multi-Profile Optimization
- [ ] Test unified profile (T=[584,592], B=[1,8])
- [ ] Benchmark profile switching overhead
- [ ] Optimize for target deployment scenario

### 7.3 Real Audio Validation
- [ ] Test on real audio samples (not random features)
- [ ] Validate against known ground truth transcriptions
- [ ] Measure WER impact (if any)

---

## Troubleshooting Guide

### Issue: Contract Assertion Failures

**Symptom:** `cache_last_channel_len_out != 0`
**Action:**
1. Inspect ONNX graph for conditional logic on cache_len
2. Re-run ORT parity to verify baseline
3. Check if engine build modified graph semantics
4. **ESCALATE:** Re-validate full cache parity before proceeding

### Issue: High encoder_output Errors (> 1e-3)

**Symptom:** Errors exceed tolerance on multiple chunks
**Action:**
1. Compare to ORT baseline (is TRT worse than ORT?)
2. Check if using FP16/INT8 (quantization expected to increase errors)
3. Inspect layer precision (mixed precision issues?)
4. Validate input data matches reference exactly

### Issue: Latency Regression

**Symptom:** TRT slower than PyTorch or ORT
**Action:**
1. Profile with `nsys` or TensorRT profiler
2. Check memory transfer overhead (minimize H2D/D2H)
3. Verify optimization profiles match workload
4. Try different precision modes (FP16 often faster)

---

## Sign-Off

After completing all phases, confirm:
- [ ] ✅ All contract assertions pass (300 chunks)
- [ ] ✅ Encoder output errors within tolerance (< 5e-4 @ p95)
- [ ] ✅ No monotonic error growth in closed-loop
- [ ] ✅ Performance meets targets (< 21ms per chunk)
- [ ] ✅ CI/CD integration complete

**Once all boxes checked:** TRT integration is production-ready.

---

**References:**
- [TRT_INTEGRATION_CLEARANCE.md](TRT_INTEGRATION_CLEARANCE.md) - Full clearance documentation
- [ONNX_PARITY_RESULTS.md](ONNX_PARITY_RESULTS.md) - ORT parity baseline
- [contracts/encoder_streaming.contract.json](contracts/encoder_streaming.contract.json) - Binding specification
- [CACHE_TIME_ROOT_CAUSE_ANALYSIS.md](CACHE_TIME_ROOT_CAUSE_ANALYSIS.md) - Known issue deep-dive
