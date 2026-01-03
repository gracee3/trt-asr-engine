# Magnolia Integration Handoff: TRT Streaming ASR

**Purpose:** Handoff for wiring TRT engines into Magnolia live captioning pipeline
**Prerequisite:** TRT integration complete (see [AGENT_SETUP_GUIDE.md](AGENT_SETUP_GUIDE.md))
**Last Updated:** 2026-01-03

---

## Mission

Wire the already-built **Parakeet streaming TRT engines** into **Magnolia** (using `parakeet_stt`, **Transcription** tile, **Audio Input** tile, and **DSP** tiles) with primary focus on:

1. **End-to-end low latency**
2. **Incremental partial/final transcript updates**
3. Later: **multi-engine fan-out** and **LLM-driven refinement**

---

## 0) What is Already Done (Do Not Redo)

**TensorRT engines are built and validated.**

| Asset | Status | Notes |
|-------|--------|-------|
| FP32 TRT engine | ✅ Built | ~2.4GB, latency target met |
| FP16 TRT engine | ✅ Built | ~1.2GB, 35% faster, 2-3x higher error |
| TRT parity harness | ✅ Validated | Passes key streaming contract invariants |
| Closed-loop stability | ✅ Demonstrated | No error accumulation trend |
| Cache mismatch issue | ✅ Resolved | Non-blocking due to `cache_len=0` contract |

**Key baseline parity results (TRT, FP32, 50 chunks):**

| Mode | Pass Rate | P95 Error | P99 Error |
|------|-----------|-----------|-----------|
| Functional | 90% | 4.88e-4 | 9.70e-4 |
| Closed-loop | 88% | 5.05e-4 | 9.20e-4 |

---

## 1) Artifacts and Where to Look

### Engines and Tools

| File | Purpose |
|------|---------|
| `out/trt_engines/encoder_streaming_fp32.plan` | FP32 production engine |
| `out/trt_engines/encoder_streaming_fp16.plan` | FP16 fast engine |
| `tools/tensorrt/trt_streaming_parity.py` | Reference implementation of bindings + correctness checks |
| `tools/tensorrt/plot_stability.py` | Error trend visualization |

### Contracts and Invariants

| File | Purpose |
|------|---------|
| `contracts/encoder_streaming.contract.json` | Binding specification |
| [AGENT_SETUP_GUIDE.md](AGENT_SETUP_GUIDE.md) | Full environment setup |

**Contract truth: chunk-isolated streaming**
- `encoded_lengths == 1`
- `encoder_output.shape[-1] == 1`
- `cache_last_channel_len_out == 0` for every chunk (decisive invariant)

---

## 2) Non-Negotiable Runtime Assertions

**MUST be enforced in Magnolia at every inference step:**

```python
# Hard-fail if any assertion fails
assert np.all(outputs["encoded_lengths"] == 1), \
    "Streaming contract violated: encoded_lengths != 1"

assert outputs["encoder_output"].shape[-1] == 1, \
    "Streaming contract violated: encoder_output time_dim != 1"

assert np.all(outputs["cache_last_channel_len_out"] == 0), \
    "Streaming contract violated: cache_len != 0. Config drift detected!"
```

**If any assertion fails:** Hard fail / block pipeline. This indicates streaming config drift and invalidates prior clearance.

---

## 3) Primary Goal: Magnolia "Live Captions" Pipeline

Implement a low-latency pipeline:

```
Audio Input tile
    ↓
DSP tiles (denoise/AGC/VAD as needed; compute or pass features)
    ↓
parakeet_stt (TRT-backed)
    ↓
Transcription tile (UI)
```

### Design Requirements

- UI must update rapidly with **partials** (hypotheses) and later **finals** (committed text)
- Displayed text must be **editable by the model over a short revision window** (to allow corrections when more context arrives) without causing distracting flicker
- Keep audio→text path non-blocking; no LLM on the critical path

---

## 4) Partials vs Finals: Implement "Stable Prefix + Revision Window"

### Transcript State Model

Implement transcript state as two regions:

| Region | Behavior |
|--------|----------|
| **Committed / final prefix** | Never changes once committed |
| **Uncommitted / partial suffix** | May be revised as new chunks arrive |

### Commitment Policy (Tunable)

- Keep last *N tokens* or *W milliseconds* uncommitted (revision window)
- Commit older text when:
  - Unchanged for K updates, OR
  - VAD/endpointing indicates boundary, OR
  - Exceeds time threshold (~1-2s behind live edge)

### Implementation Guidance

On each decoding update:
1. Compute diff vs previous hypothesis (token-based LCP or edit-distance)
2. Only patch UI for the changed suffix
3. Never rewrite committed prefix (unless explicitly supporting "retro-corrections")

### User Expectation

Users tolerate changes in last ~1-2 seconds; they dislike changes earlier. Design UI accordingly.

---

## 5) Low-Latency Engineering Constraints

To preserve the ~12ms (FP16) / ~18ms (FP32) engine advantage:

| Requirement | Implementation |
|-------------|----------------|
| Minimize buffering | No large buffers in Audio Input / DSP |
| Avoid GPU waits on UI thread | Use async/non-blocking GPU calls |
| Buffer reuse | Pre-allocate bindings, reuse buffers |
| Fast memory | Use pinned host memory for H2D/D2H copies |
| Dedicated inference | Use separate inference thread + UI queue |

### Latency Measurement

Measure **end-to-end latency** as:

```
(audio timestamp of last sample consumed) → (time UI renders updated text)
```

---

## 6) Multi-Engine Roadmap (Not v1; Design for It)

### Later Goal

Run **3-4 engines** on same audio, produce "best" transcript, optionally feeding an LLM.

### Plan for Extensibility Now

| Component | Purpose |
|-----------|---------|
| **Audio/features fan-out bus** | Multiple recognizers subscribe without duplicating DSP |
| **Async engine lanes** | Each engine in its own async lane |
| **Hypothesis aggregator interface** | Pluggable aggregation strategy |

### Aggregator Evolution

| Version | Behavior |
|---------|----------|
| v1 | Single engine only |
| v2 | Fast engine drives UI, slower engine proposes corrections |
| v3 | N-engine arbitration and/or LLM selection |

### Multi-Engine Best Practice

- **Engine A (fastest)** drives **live captions** (low latency)
- Engines B/C/D run in background, propose corrections **only inside revision window**
- Avoids "wait for consensus" stall

### Optimization: Shared Encoder

If architecture allows:
- **One encoder** (shared)
- Multiple decoders (different beams / LMs / rescoring)

Saves GPU memory and reduces contention.

---

## 7) LLM Integration Rules (Async Only)

**LLM must not block captions.** Use it for:

- Punctuation/casing
- Selecting among competing engine hypotheses
- Higher-level corrections or summaries

### LLM Guardrails

- Do NOT hallucinate words not present in ASR hypotheses
- Prefer "choose/merge from candidates" prompts over "rewrite freely"

### Operational Pattern

1. Stream **finalized segments** (stable prefix chunks) to LLM
2. LLM returns "polished transcript" view that updates behind raw captions
3. If LLM updates same text user is reading, restrict to:
   - Punctuation/casing-only edits, OR
   - Edits only within revision window

---

## 8) Definition of Done (v1: Single Engine Captions)

- [ ] Captions update continuously with partials
- [ ] Finals emitted on endpoint (VAD) or stable prefix threshold
- [ ] All three runtime assertions enforced and always pass
- [ ] End-to-end latency measured and reported; no regressions from buffering
- [ ] Recorded demo shows responsive live transcription

---

## 9) Recommended Implementation Order

### Phase 1: Single-Engine Wiring

1. Wire: Audio → DSP → parakeet_stt(TRT) → Transcription tile
2. Implement transcript state with:
   - Committed prefix
   - Revisable suffix (windowed)
3. Add endpointing (VAD) to emit finals
4. Measure and optimize latency

### Phase 2: Refinement Lane

5. Add second engine as "refinement lane"
6. Implement hypothesis aggregator (v2)
7. Test correction latency vs UI responsiveness

### Phase 3: LLM Integration

8. Add async LLM for punctuation/casing
9. Implement hypothesis selection prompt
10. Add polished transcript view

---

## UI Implementation Notes

### Minimal Patching Strategy (Critical for Responsiveness)

Avoid re-rendering whole transcript each update:

```python
# State model
class TranscriptState:
    committed_text: str  # Never changes once set
    partial_text: str    # May change each update

# Render: committed_text + partial_text
# On update: only patch partial span
```

Even with 12ms engine, you lose that advantage with heavy UI/string rebuild work every tick.

### Diff-Based Updates

```python
def update_partial(old_partial: str, new_partial: str) -> Patch:
    # Compute minimal edit to transform old → new
    # Return patch for UI to apply (not full re-render)
    pass
```

---

## References

| Document | Purpose |
|----------|---------|
| [AGENT_SETUP_GUIDE.md](AGENT_SETUP_GUIDE.md) | Environment, paths, commands |
| [TRT_INTEGRATION_CHECKLIST.md](TRT_INTEGRATION_CHECKLIST.md) | Full integration history |
| [contracts/encoder_streaming.contract.json](contracts/encoder_streaming.contract.json) | I/O binding specification |
| [tools/tensorrt/trt_streaming_parity.py](tools/tensorrt/trt_streaming_parity.py) | Reference TRT bindings |

---

**Handoff Created:** 2026-01-03
**Prerequisites:** TRT engines validated, parity tests passing
**Next Milestone:** Single-engine Magnolia wiring with partials/finals
