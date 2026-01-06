# Restart Plan: trt-asr-engine (Parakeet-TDT)

Goal: methodical, paper-aligned rebuild of the full pipeline (NeMo -> ONNX -> ORT parity -> TRT engines -> low-latency C++ runtime + Rust FFI) with strict, source-backed contracts.

## Ground rules
- Treat all existing implementation logic and constants as speculative unless backed by paper, NeMo config, or export metadata.
- Python is for export, validation, diagnostics, and CI only.
- Runtime hot path stays in C++ + Rust and must be allocation-minimal.
- Keep decision log updated: `docs/DECISION_LOG.md`.

## Phase A: Filesystem scan + inventory
**Objective:** understand what exists before changing anything.

Steps:
- Capture tree: `tree -L 5 > docs/inventory/tree.txt`.
- Enumerate build systems and tool entrypoints (see `docs/inventory/INVENTORY.md`).
- Inventory contracts, metadata, parity tools, instrumentation toggles.

Deliverable:
- `docs/inventory/INVENTORY.md`.

Gate A:
- Inventory doc lists repo map, tooling map, runtime map, current contracts, unknowns.

## Phase B: Source-backed contract
**Objective:** define a single contract that is paper- and metadata-backed.

Steps:
- Extract NeMo config from `.nemo` and record the values.
- Use `tools/export_onnx/out/model_meta.json` as export metadata baseline.
- Map each contract field to sources in `docs/CONTRACT_SOURCES.md`.
- Draft unified contract: `contracts/parakeet-tdt-0.6b-v3.contract.json`.

Deliverables:
- `docs/CONTRACT_SOURCES.md`.
- `contracts/parakeet-tdt-0.6b-v3.contract.json`.

Gate B:
- Every numeric contract value has a source in `docs/CONTRACT_SOURCES.md`.
- Gaps are explicitly listed with follow-up actions.

## Phase C: Deterministic ONNX export
**Objective:** re-export encoder/predictor/joint (and streaming if applicable) with deterministic metadata.

Commands:
```bash
python -u tools/export_onnx/export.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out out/onnx/parakeet-tdt-0.6b-v3/base \
  --component all \
  --device cpu \
  --smoke-test-ort

python -u tools/export_onnx/export.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --out out/onnx/parakeet-tdt-0.6b-v3/streaming \
  --component encoder_streaming \
  --streaming-cache-size 256 \
  --streaming-cache-drop-size 0 \
  --device cpu
```

Deliverable:
- `docs/EXPORT_REPORT.md` (export commands, versions, hash list, artifacts).

Gate C:
- ONNX graphs pass `onnx.checker`.
- `model_meta.json` reflects the new export and matches the contract.

## Phase D: ORT parity ladder
**Objective:** prove PyTorch -> ORT parity per component and end-to-end decode.

Commands (example):
```bash
# PyTorch reference JSONL
python tools/verify_nemo/streaming_encoder_reference.py \
  --model models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo \
  --device cuda \
  --cache-size 256 \
  --chunk-len 592 \
  --num-chunks 50 \
  --seed 42 \
  --jsonl-out artifacts/reference/pytorch_reference_50.jsonl

# ORT parity (streaming encoder)
python tools/onnxruntime/onnx_streaming_parity.py \
  --onnx out/onnx/parakeet-tdt-0.6b-v3/streaming/encoder_streaming.onnx \
  --ref artifacts/reference/pytorch_reference_50.jsonl \
  --mode functional \
  --providers cuda
```

Deliverable:
- `docs/VALIDATION_REPORT_ORT.md`.

Gate D:
- Encoder, predictor, joint parity within tolerances.
- Decode parity (token IDs) matches PyTorch on at least one deterministic test.

## Phase E: TensorRT build + TRT parity
**Objective:** build engines and prove TRT parity vs ORT/PyTorch.

Commands:
```bash
python tools/build_trt/build_trt.py \
  --meta out/onnx/parakeet-tdt-0.6b-v3/base/model_meta.json \
  --outdir models/parakeet-tdt-0.6b-v3 \
  --fp16
```

Deliverable:
- `docs/VALIDATION_REPORT_TRT.md`.

Gate E:
- FP32 TRT parity within tolerance.
- FP16 tolerance is quantified and argmax stability is measured.

## Phase F: Runtime rebuild (C++ + Rust)
**Objective:** implement contract-accurate, low-latency runtime.

Steps:
- Implement contract assertions at every call boundary.
- Implement TDT greedy decode with max-symbol guard.
- Wire Rust features to C++ core via thin FFI.
- Add microbenchmark harness and latency measurements.

Deliverable:
- `docs/ARCHITECTURE_RUNTIME.md`.

Gate F:
- Runtime enforces contract assertions.
- No heap allocations in steady-state loop.
- End-to-end streaming latency measured and meets budget.

## Phase G: Repo organization proposal
**Objective:** create a clean, maintainable repo layout for long-term work.

Deliverable:
- `docs/REPO_ORGANIZATION_PROPOSAL.md`.

Gate G:
- Migration plan preserves current tools and supports incremental moves.

## Acceptance gates (stop/go)
- Gate 1: Contract complete + source-backed.
- Gate 2: ORT parity passes for components + decode.
- Gate 3: TRT parity passes with no drift.
- Gate 4: Runtime correct, observable, and fast.

## Decision logging standard
Any discretionary value in code must have a contract-style comment:
```cpp
// CONTRACT: <what invariant is being enforced>
// SOURCE: <paper section / NeMo config key / export meta field / doc path>
// WHY: <why this matters to parity or latency>
// TODO: <if uncertain, what experiment will validate it>
```

Maintain `docs/DECISION_LOG.md` for every decision.
