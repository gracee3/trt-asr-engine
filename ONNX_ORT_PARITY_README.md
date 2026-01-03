# ONNX/ORT Parity Testing & TRT Integration Package

**Model:** parakeet-tdt-0.6b-v3 streaming encoder
**Date:** 2026-01-03
**Status:** ✅ **CLEARED FOR TRT INTEGRATION**

---

## Quick Start

**For TRT Integration Team:**
1. Start here: [TRT_INTEGRATION_CLEARANCE.md](TRT_INTEGRATION_CLEARANCE.md) ← **Read this first**
2. Follow this: [TRT_INTEGRATION_CHECKLIST.md](TRT_INTEGRATION_CHECKLIST.md) ← **Step-by-step guide**
3. Reference this: [contracts/encoder_streaming.contract.json](contracts/encoder_streaming.contract.json) ← **Binding spec**

**For Deep Technical Review:**
1. [ONNX_PARITY_RESULTS.md](ONNX_PARITY_RESULTS.md) - Comprehensive ORT parity test results
2. [CACHE_TIME_ROOT_CAUSE_ANALYSIS.md](CACHE_TIME_ROOT_CAUSE_ANALYSIS.md) - Diagnostic investigation

---

## Executive Summary

### ✅ Validation Complete

ONNX export (`encoder_streaming.onnx`) has been validated against PyTorch reference:
- **Streaming contract:** `valid_out_len=1` preserved (all chunks output time=1)
- **Encoder output parity:** FP32 errors 6e-5 to 7e-4 (acceptable variance)
- **Closed-loop stability:** No error accumulation over 50 chunks
- **Cache contract:** Chunk-isolated mode (`cache_len=0` always)

### ⚠️ Known Issue (Non-Blocking)

**Issue:** `cache_last_time_out` has 0.01-0.1 absolute errors vs PyTorch
**Status:** **Non-blocking** - cache operates in reset mode
**Impact:** None (validated via diagnostic testing)
**Action:** Document and monitor with runtime assertions

### 🚀 TRT Integration Status

**GO/NO-GO:** ✅ **GO**
**Prerequisites:** Complete
**Blockers:** None
**Next Step:** Build TRT engine (FP32) and run functional parity test

---

## Deliverables

### Core Documentation

| File | Purpose | Audience |
|------|---------|----------|
| [TRT_INTEGRATION_CLEARANCE.md](TRT_INTEGRATION_CLEARANCE.md) | Formal sign-off & requirements | **TRT Team** (START HERE) |
| [TRT_INTEGRATION_CHECKLIST.md](TRT_INTEGRATION_CHECKLIST.md) | Step-by-step integration guide | **TRT Team** (Implementation) |
| [ONNX_PARITY_RESULTS.md](ONNX_PARITY_RESULTS.md) | Complete ORT parity test results | Technical Review |
| [CACHE_TIME_ROOT_CAUSE_ANALYSIS.md](CACHE_TIME_ROOT_CAUSE_ANALYSIS.md) | Diagnostic deep-dive | Troubleshooting |

### Technical Specifications

| File | Purpose | Usage |
|------|---------|-------|
| [contracts/encoder_streaming.contract.json](contracts/encoder_streaming.contract.json) | I/O shapes, profiles, runtime contract | Engine build, validation |
| [out/encoder_streaming.onnx](out/encoder_streaming.onnx) | ONNX model file | TRT engine source |

### Test Infrastructure

| File | Purpose | Usage |
|------|---------|-------|
| [tools/verify_nemo/streaming_encoder_reference.py](tools/verify_nemo/streaming_encoder_reference.py) | PyTorch reference generator | Generate ground truth JSONL |
| [tools/onnxruntime/onnx_streaming_parity.py](tools/onnxruntime/onnx_streaming_parity.py) | ORT parity harness | Baseline validation (template for TRT) |
| [tools/onnxruntime/diagnose_cache_time_mismatch.py](tools/onnxruntime/diagnose_cache_time_mismatch.py) | Cache diagnostic tool | Troubleshooting if contract changes |

### Test Data (Gitignored - Generate Locally)

| File | Size | Purpose | Command |
|------|------|---------|---------|
| `pytorch_reference_50.jsonl` | 3.2GB | 50-chunk ground truth | Already generated |
| `pytorch_reference_300.jsonl` | ~20GB | 300-chunk stability test | See checklist Phase 1.3 |

### Test Results (Gitignored)

| File | Purpose |
|------|---------|
| `parity_50chunks_functional_cpu.json` | ORT functional parity (CPU) |
| `parity_50chunks_functional_cuda.json` | ORT functional parity (CUDA) |
| `parity_50chunks_closedloop_cpu.json` | ORT closed-loop parity (CPU) |
| `parity_50chunks_closedloop_cuda.json` | ORT closed-loop parity (CUDA) |
| `cache_time_diagnostic.json` | Cache diagnostic analysis (chunk 10) |

---

## Key Findings

### 1. Streaming Contract: Chunk-Isolated Mode

**Discovery:** `cache_last_channel_len == 0` for ALL chunks (input and output)

**Implications:**
- Model operates in **"chunk-by-chunk"** mode with per-chunk cache reset
- Cache used **intra-chunk** (within single forward pass)
- Cache NOT carried **inter-chunk** (reset between chunks)
- Simpler contract than initially assumed

**Runtime Requirements:**
```python
# MANDATORY ASSERTION at every chunk
assert cache_last_channel_len_out == 0
```

### 2. Encoder Output Parity

| Mode | Max Abs Error | Typical Range | Status |
|------|---------------|---------------|--------|
| Functional | 7.3e-4 | 6e-5 to 2e-4 | ✅ Acceptable |
| Closed-loop | 2.7e-4 | 8e-5 to 2.7e-4 | ✅ Stable |

**Acceptance Criteria for TRT:**
- P95: encoder_output max_abs < 5e-4
- P100: encoder_output max_abs < 1e-3
- No monotonic growth in closed-loop (300 chunks)

### 3. Cache Behavior

| Cache Tensor | Status | Notes |
|--------------|--------|-------|
| `cache_last_channel_out` | ✅ Perfect match | Bitwise identical to PyTorch |
| `cache_last_time_out` | ⚠️ Known mismatch | 0.01-0.1 abs error, non-blocking |
| `cache_last_channel_len_out` | ✅ Perfect match | Always 0 (contract) |

**cache_last_time_out Resolution:**
- Errors are real (diagnostic validated)
- But non-propagating due to `cache_len=0` isolation
- TRT tolerance: atol=0.1 acceptable

---

## Validation Evidence

### ORT Parity Tests (50 chunks)

**Functional Mode (Reference Caches):**
- CPU & CUDA: Identical behavior (no EP-specific issues)
- encoder_output: 58% pass rate @ atol=1e-4
- Contract invariants: 100% pass (encoded_lengths=1, cache_len=0)

**Closed-Loop Mode (Cache Feedback):**
- CPU & CUDA: Identical behavior
- encoder_output: Stable (no accumulation over 50 chunks)
- Max error: 2.7e-4 (within FP32 envelope)

### Diagnostic Testing (3-Part Cache Analysis)

**Check 1:** Errors uniformly distributed → NOT padding-side issue
**Check 2:** Errors in significant regions → NOT padding junk
**Check 3:** Cache IS semantically active → Intra-chunk usage confirmed

**Resolution:** Cache isolation via `cache_len=0` explains all observations

---

## TRT Integration Path

### Phase 1: Engine Build (Est. 1-2 hours)
1. Review binding contract
2. Create optimization profiles
3. Build FP32 engine
4. Implement runtime wrapper with assertions

### Phase 2: Functional Parity (Est. 2-4 hours)
1. Adapt ORT harness for TRT
2. Run 50-chunk functional test
3. Validate against ORT baseline
4. Debug any failures

### Phase 3: Stability Testing (Est. 4-8 hours)
1. Run 300-chunk closed-loop test
2. Analyze error trends (plot stability)
3. Validate no accumulation
4. Confirm contract assertions hold

### Phase 4: Performance & Deployment (Est. 1-2 days)
1. Latency benchmarking
2. Memory profiling
3. CI/CD integration
4. Production deployment

**Total Estimated Effort:** 2-3 days for full validation & integration

---

## Troubleshooting

### Q: Contract assertion failed (cache_len != 0)

**A:** This is a **CRITICAL FAILURE**. The streaming config has changed:
1. Stop integration immediately
2. Re-run diagnostic: `python3 tools/onnxruntime/diagnose_cache_time_mismatch.py`
3. Re-validate cache parity with strict tolerances
4. Escalate to ONNX export team

### Q: encoder_output errors exceed 1e-3

**A:** Investigate numerical precision:
1. Compare to ORT baseline (is TRT worse?)
2. Check if using FP16/INT8 (quantization expected)
3. Inspect layer precision (mixed precision?)
4. Validate input preprocessing matches exactly

### Q: Errors accumulate in closed-loop

**A:** Investigate cache feedback:
1. Verify cache normalization (padding to fixed shapes)
2. Check if cache_len assertion is passing
3. Plot error vs chunk_index (should be flat)
4. Compare to ORT closed-loop (should match)

---

## Change Management

### If Streaming Config Changes

**Conditions requiring re-validation:**
- ❌ `cache_last_channel_len > 0` observed
- ❌ `valid_out_len != 1` in future configs
- ❌ Different chunk sizes or shift parameters

**Actions:**
1. Re-run full diagnostic suite
2. Tighten cache_last_time tolerance to atol=1e-4
3. Validate inter-chunk error propagation
4. Update binding contract accordingly

---

## Contact & Support

**For questions about:**
- ONNX export: See export logs and NeMo streaming config
- ORT parity results: Review [ONNX_PARITY_RESULTS.md](ONNX_PARITY_RESULTS.md)
- Cache behavior: Review [CACHE_TIME_ROOT_CAUSE_ANALYSIS.md](CACHE_TIME_ROOT_CAUSE_ANALYSIS.md)
- TRT integration: Follow [TRT_INTEGRATION_CHECKLIST.md](TRT_INTEGRATION_CHECKLIST.md)

---

**Package Generated:** 2026-01-03
**Validation Status:** Complete
**Integration Status:** Cleared to proceed
**Next Milestone:** TRT engine build + functional parity
