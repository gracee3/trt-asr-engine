# Inventory (Phase A)

This is a repo scan snapshot for trt-asr-engine. Tree view captured in `docs/inventory/tree.txt`.

## Repo map (high level)
- `apps/`: small app entrypoints (ex: `apps/trt_runtime_smoke/`).
- `cpp/`: TensorRT runtime + C ABI + decoder logic.
- `rust/`: feature extraction, FFI, CLI.
- `tools/`: Python tooling (export, parity, TRT build, diagnostics, STT suite).
- `contracts/`: machine-readable binding contracts.
- `docs/`: human-readable docs.
- `eval/`: pinned evaluation manifests (WAVs are generated/ignored).
- `models/`: local model artifacts (`.nemo`, built engines).
- `out/`: generated ONNX exports and external data (gitignored).
- `artifacts/`: generated parity/stability outputs (gitignored).
- `debug_artifacts/`: runtime tap outputs and debug data.

## Build systems detected
- CMake: `cpp/CMakeLists.txt`, `apps/trt_runtime_smoke/CMakeLists.txt`.
- Cargo: `rust/Cargo.toml` and per-crate `Cargo.toml` files.
- Python requirements: `tools/export_onnx/requirements.txt`, `tools/verify_nemo/requirements.txt`.

## Tooling map (entrypoints under `tools/`)
- Export:
  - `tools/export_onnx/export.py` (NeMo -> ONNX; writes `model_meta.json`).
  - `tools/export_onnx/export_nemo.py` (NeMo export helper).
- NeMo verification:
  - `tools/verify_nemo/verify.py` (golden transcript check).
  - `tools/verify_nemo/streaming_encoder_reference.py` (PyTorch reference JSONL).
  - `tools/verify_nemo/streaming_encoder_cache.py` (streaming cache sanity).
  - `tools/verify_nemo/audit_model_arch.py` (architecture audit).
- ONNX Runtime parity:
  - `tools/onnxruntime/onnx_streaming_parity.py` (functional + closed loop).
  - `tools/onnxruntime/diagnose_cache_time_mismatch.py` (cache mismatch diagnostics).
- ONNX inspection:
  - `tools/inspect_onnx/check_joint_output.py` (verifies joint output shape + absence of LogSoftmax).
- TensorRT parity:
  - `tools/tensorrt/trt_streaming_parity.py` (TRT parity harness).
  - `tools/tensorrt/plot_stability.py` (error drift plots).
- TRT build:
  - `tools/build_trt/build_trt.py` (trtexec wrapper).
  - `tools/build_trt/scripts/build_all.sh` and `tools/build_trt/scripts/inspect_engine.py`.
- Debugging:
  - `tools/analyze_tap.py` (audio/feature tap analysis).
- STT test suite:
  - `tools/stt_suite/run_suite.py` (CLI + loopback suite).
  - `tools/stt_suite/make_manifest.py` and `tools/stt_suite/make_librispeech_manifest.py`.
  - `tools/stt_suite/make_gate_manifest.py` (pinned dev-clean/dev-other gate manifest).
  - `tools/stt_suite/score_wer.py` and `tools/stt_suite/run_librispeech_loopback_suite.sh`.

## Validation + parity harnesses
- ORT parity: `tools/onnxruntime/onnx_streaming_parity.py` (functional, closed-loop).
- TRT parity: `tools/tensorrt/trt_streaming_parity.py` (functional, closed-loop).
- PyTorch reference data: `tools/verify_nemo/streaming_encoder_reference.py` writes JSONL (base64 arrays).
- Diagnostics: `tools/onnxruntime/diagnose_cache_time_mismatch.py`.
- Existing results and guidance:
  - `ONNX_ORT_PARITY_README.md`, `ONNX_PARITY_RESULTS.md`.
  - `TRT_INTEGRATION_CHECKLIST.md`, `TRT_INTEGRATION_CLEARANCE.md`.
  - `CACHE_TIME_ROOT_CAUSE_ANALYSIS.md`.

## Instrumentation + runtime toggles
- Audio taps:
  - `AUDIO_TAP_ENABLE`, `AUDIO_TAP_DIR`, `AUDIO_TAP_CAPTURE`, `AUDIO_TAP_POST_DSP`, `AUDIO_TAP_FEATURES`.
  - Implemented in `cpp/include/audio_tap.h` and referenced in `cpp/src/parakeet_trt.cpp`.
- NaN/Inf guards:
  - `PARAKEET_NAN_GUARD_ALWAYS`, `PARAKEET_NAN_GUARD_HALT`.
- Decode debug:
  - `PARAKEET_DEBUG_BLANK_SCAN`, `PARAKEET_DEBUG_EMIT_TOKENS`, `PARAKEET_DEBUG_JOINT_TOPK`.
  - `PARAKEET_BLANK_PENALTY`, `PARAKEET_JOINT_DUR_FIRST`, `PARAKEET_Y0_OVERRIDE`.
- Cache controls:
  - `PARAKEET_CACHE_LEN_OVERRIDE`, `PARAKEET_DISABLE_CACHE`, `PARAKEET_MAX_FRAMES_PER_PUSH`.
- Perf + timing:
  - `PARAKEET_DEBUG_DEVICE_SYNC`, `PARAKEET_DEBUG_SYNC_MEMCPY`, `PARAKEET_SLOW_MEMCPY_MS`.
  - `PARAKEET_SLOW_CHUNK_MS`, `PARAKEET_SLOW_ENQUEUE_MS`.
  - Slot reuse toggles: `PARAKEET_SLOT_REUSE_*`.

## Runtime map
- C++ core:
  - `cpp/src/parakeet_trt.cpp`: engine load, streaming loop, decode, instrumentation.
  - `cpp/src/engine_manager.*`: engine lifecycle and binding management.
  - `cpp/src/decoder.*`: greedy TDT decode helper (token+duration heads).
  - `cpp/include/audio_tap.h`: reusable tap writer.
- Rust:
  - `rust/features/`: log-mel feature extraction.
  - `rust/parakeet_trt_sys/`: C ABI bindings.
  - `rust/parakeet_trt/`: higher-level Rust wrapper.
  - `rust/cli/`: CLI entrypoint.

## Contracts + metadata sources
- `contracts/encoder_streaming.contract.json`: legacy streaming encoder binding contract (chunk-isolated, layer-first).
- `docs/runtime_contract.md`: offline encoder/predictor/joint bindings.
- `tools/export_onnx/out/model_meta.json` (also `out/model_meta.json`): export metadata.
- `audit_model_arch.json`: architecture audit from NeMo checkpoint.
- `models/parakeet-tdt-0.6b-v3/build_report.json`: TRT build profiles/params.

## Unknowns / decision points
- Streaming mode target is stateful cache carryover; ORT closed-loop parity now passes with batch-first cache layout.
- Export cache size mismatch: `last_channel_cache_size=10000` (NeMo config) vs `cache_size=256` (streaming export).
- Blank + duration=0 policy: paper disallows; runtime currently has a heuristic.
- Max symbols per timestep: runtime uses 8 (speculative; needs source or experiment).
- Predictor details: RNN cell type not explicit in config; confirm from NeMo.
- Feature normalization: `normalize=per_feature` uses per-utterance mean/std (not streaming-safe); decide whether to keep model-matching or override.
- Full pipeline parity harness: only streaming encoder parity exists; predictor/joint/decode parity still needed.
