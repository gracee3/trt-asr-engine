# Runtime Architecture (C++ core + Rust edge)

## Overview
The runtime is a low-latency streaming ASR engine for Parakeet-TDT. The hot path is in C++ (TensorRT execution + decode loop). Rust provides feature extraction, CLI, and a thin FFI layer.

```
Audio -> Rust features (log-mel) -> C++ TRT encoder/predictor/joint -> TDT greedy decode -> C ABI -> Rust CLI
```

## C++ core (hot path)
Responsibilities:
- Load TensorRT engines for encoder, predictor, and joint.
- Pre-allocate device/host buffers and reuse them across chunks.
- Execute TRT with async CUDA streams (avoid device sync in hot path).
- Maintain streaming state (encoder caches + predictor state + decode position).
- Implement TDT greedy decode with guardrails (max symbols per timestep, blank policy).
- Emit partial/final transcripts via the C ABI.
- Expose instrumentation (taps, NaN guards, decode logs).

Key modules:
- `cpp/src/engine_manager.*`: engine load + bindings map.
- `cpp/src/parakeet_trt.cpp`: streaming loop, decode, instrumentation.
- `cpp/src/decoder.*`: token/duration argmax helper.
- `cpp/include/audio_tap.h`: tap writer for audio/features.

## Rust edge
Responsibilities:
- Feature extraction (log-mel) that matches the contract.
- Audio I/O (wav/PCM) for CLI and integration harnesses.
- Thin FFI boundary with minimal per-chunk overhead.
- Replay harness for taps (deterministic debugging).

Key crates:
- `rust/features/`: log-mel extraction.
- `rust/parakeet_trt_sys/`: C ABI bindings.
- `rust/parakeet_trt/`: safe wrapper.
- `rust/cli/`: CLI entrypoint.

## Data flow and memory
- Inputs are batched as `B=1` (streaming).
- Features are `float32` on host; copied to device once per chunk.
- TRT outputs remain on device until a small CPU copy is required for decode (token/duration argmax).
- All buffers are allocated once at session init; no heap allocation in the steady-state loop.

## Streaming state model
Single struct (per session) to carry all mutable state:
- Encoder caches:
  - `cache_last_channel` [L, B, C, D]
  - `cache_last_time` [L, B, D, K]
  - `cache_last_channel_len` [B]
  - Cache tensors are carried across chunks; `cache_last_channel_len` tracks valid cache depth.
- Predictor state:
  - `h`/`c` [L, B, H]
- Decode state:
  - current encoder time index `t`
  - token history/hypothesis buffer
- Chunking buffers:
  - feature ring buffer
  - encoder output ring buffer

## TDT greedy decode (contract-aligned)
- Slice joint output into token logits and duration logits.
- `best_tok = argmax(token_logits)`.
- `best_dur = argmax(duration_logits)`.
- If `best_tok` is non-blank, append token and advance predictor state.
- Advance encoder time by duration (`duration_values[best_dur]`).
- Guard: `max_symbols_per_timestep` to avoid infinite loops.
- Blank + duration=0 must be handled per contract (disallow or override).
- Joint output is exported as raw logits; apply per-head softmax only when probabilities are needed.

## FFI boundary
- C ABI in `cpp/include/parakeet_trt.h`.
- Keep FFI calls coarse-grained (per chunk), not per frame.
- Avoid per-call allocations and string churn; reuse buffers and pass lengths.

## Instrumentation and test hooks
- Audio/feature taps: `AUDIO_TAP_*` env vars and `audio_tap.h`.
- NaN guards: `PARAKEET_NAN_GUARD_*`.
- Decode debug logs: `PARAKEET_DEBUG_*`.
- Parity harnesses:
  - `tools/onnxruntime/onnx_streaming_parity.py`
  - `tools/tensorrt/trt_streaming_parity.py`
- Replay harness in Rust CLI for deterministic reproduction.

## Performance constraints
- No heap allocations in the steady-state streaming loop.
- No CPU<->GPU transfers beyond minimal logits for argmax.
- One CUDA stream per session; avoid device-wide sync.
- Maintain bounded latency: measure end-to-end (last audio sample -> text output).

## Contract enforcement
The runtime must hard-fail on contract violations (shape, dtype, cache semantics). See `contracts/parakeet-tdt-0.6b-v3.contract.json` and `docs/runtime_contract.md`.
