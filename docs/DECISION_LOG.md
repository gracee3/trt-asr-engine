# Decision Log

## 2026-01-06
- Decision: Use a single JSON contract (`contracts/parakeet-tdt-0.6b-v3.contract.json`) as the canonical runtime contract.
  Alternatives: TOML or YAML.
  Evidence: Requirement in reset plan; JSON matches existing tooling output (`model_meta.json`).
  Validation: Schema check + parity harness uses this contract.

- Decision: Default joint layout is token-first, duration-last.
  Alternatives: duration-first layout via runtime switch.
  Evidence: NeMo TDT loss slices `acts[..., :-n_durations]` vs `acts[..., -n_durations:]` in `nemo/collections/asr/losses/rnnt_pytorch.py`; `model_config.yaml` sets `joint.num_extra_outputs=5` and `loss.tdt_kwargs.durations=[0,1,2,3,4]`; `tools/export_onnx/out/joint.onnx` output is a single linear of size 8198 (no concat).
  Validation: joint re-exported to raw logits (no LogSoftmax); output node is linear/add with size 8198; layout consistent with durations appended at tail.

- Decision: Standardize streaming cache interface as batch-first, fixed-size padded caches + explicit `cache_last_channel_len_out` valid length.
  Alternatives: layer-first cache layout or ragged cache tensors.
  Evidence: NeMo `input_types_for_export` uses batch-first cache tensors; ORT parity aligns when caches are padded to `cache_size` and valid length is used to compare.
  Validation: closed-loop ORT parity passes for 4 chunks with `cache_last_channel_len_out` monotonic and cache shape alignment.

- Decision: Decode guard `max_symbols_per_timestep=8` is provisional.
  Alternatives: derive from paper or make configurable from contract.
  Evidence: current runtime constant in `cpp/src/parakeet_trt.cpp`.
  Validation: experiment with repetition-heavy inputs; tune to avoid infinite loops without hurting accuracy.

- Decision: Blank + duration=0 is disallowed in decode (policy aligns with TDT paper).
  Alternatives: allow duration=0 for blank or clamp to 1.
  Evidence: TDT paper note disallowing duration=0 for blank (page 3/23).
  Validation: run decode parity vs PyTorch with both policies on deterministic samples.

- Decision: Export joint output as raw logits (no global LogSoftmax).
  Alternatives: keep LogSoftmax in ONNX and renormalize per head at runtime.
  Evidence: TDT paper specifies independently normalized token/duration heads; NeMo TDT loss slices token vs duration outputs; export now forces `joint.log_softmax=False`.
  Validation: re-export `joint.onnx` and confirm no LogSoftmax node on `joint_output`.

- Decision: Target true stateful streaming with cache carryover.
  Alternatives: keep chunk-isolated streaming (`cache_len_out == 0`).
  Evidence: streaming paper requires cache/context discipline for equivalence; NeMo `ConformerEncoder` implements cache carryover when `cache_last_channel_len` is nonzero.
  Validation: re-export streaming encoder, generate reference JSONL with nonzero cache_len, and pass closed-loop parity with cache feedback (ORT).

- Decision: Initial clamp `cache_drop_size` to 71 for model-default streaming config.
  Alternatives: accept negative cache_len or disable `drop_extra_pre_encoded`.
  Evidence: pre-encode length for chunk0 (577/584 frames) is 73; after `drop_extra_pre_encoded=2`, max usable length is 71; `cache_drop_size=72` yields negative cache_len.
  Validation: For default chunking, clamp prevents negative lengths, but with 48‑frame runtime chunks cache_len remained negative/zero; superseded by 2026-01-08 override to enable stateful caching at low latency.

- Decision: Record feature normalization semantics as per-utterance mean/std over time (`normalize=per_feature`), and flag as not streaming-safe.
  Alternatives: override with streaming-safe normalization (e.g., none or fixed stats).
  Evidence: NeMo `normalize_batch` computes mean/std across the full `seq_len` time axis per feature; streaming paper avoids mel-spectrogram normalization requiring whole-utterance stats.
  Validation: decide whether to keep model-matching normalization or override; run feature parity vs NeMo either way.

- Decision: Define timebase as 10ms feature frame shift and 8x subsampling → 80ms encoder step; TDT durations advance encoder steps.
  Alternatives: treat duration as feature-frame steps (incorrect for 8x subsampling).
  Evidence: model config `subsampling_factor=8`; Fast Conformer paper notes 10ms → 80ms; TDT algorithm advances encoder time index by duration.
  Validation: check decode timestamps and duration behavior against PyTorch reference on deterministic samples.

## 2026-01-07
- Decision: Feature normalization default remains UNLOCKED pending WER gate; both normalization modes currently yield mostly empty transcripts.
  Alternatives: lock to model-matching (`per_feature`) or streaming-safe (`none`).
  Evidence: `docs/VALIDATION_REPORT_WER.md` shows WER 100% (none) and 98.23% (per_feature) on the pinned dev gate.
  Validation: fix decode/output empties and re-run the gate to decide the default.

- Decision: TDT greedy decode advances encoder time by predicted duration for every step; blank+duration=0 is clamped to advance=1.
  Alternatives: RNNT-style blank-only advance or renormalized duration head for blank.
  Evidence: TDT Algorithm 2 advances time by duration each step; paper disallows blank+duration=0 without renormalization.
  Validation: enable `PARAKEET_DEBUG_TDT_STEPS` and confirm time_idx progression + non-empty token emissions on pinned dev-clean utterances.

## 2026-01-08
- Decision: Override streaming config for stateful caching at 48‑frame chunks (`chunk_size=[41,48]`, `shift_size=[17,24]`, `cache_drop_size=3`, `valid_out_len=3`).
  Alternatives: keep model-default streaming config (`cache_drop_size=71`, `valid_out_len=2`) which yields `cache_len_out<=0` for 48‑frame chunks.
  Evidence: `tools/verify_nemo/streaming_encoder_cache.py --chunk-size 48 --cache-drop-size 3` yields `cache_len_out=1` on chunk 0; ORT cache sensitivity shows positive cache_len_out; ORT closed-loop parity passes for 4 chunks with cache_len_out monotonic.
  Validation: re-export `encoder_streaming.onnx` with overrides, run `tools/onnxruntime/onnx_streaming_parity.py` closed-loop parity (PASS), and confirm non-negative cache_len_out across chunks.

- Decision: Expand streaming encoder TRT `audio_signal.T` profile to include `57` (chunk_size 48 + pre_encode 9), and update contract `audio_signal.T` typical values to `[41,57]`.
  Alternatives: keep `T=48` profiles or disable pre-encode (would diverge from `streaming_cfg` behavior).
  Evidence: `artifacts/reference/stream_ref_cache3_50.jsonl` metadata shows `chunk_len=57` for chunks ≥1 (`slice_end - slice_start = 48 + 9`); TRT parity with `T=41/48` profiles fails shape checks for `T=57`.
  Validation: rebuild TRT encoder with `T=41/57` profiles and re-run closed-loop parity against the cache3 reference.

- Decision: Disable TF32 for FP32 streaming encoder TRT builds to eliminate cache_last_time_out parity outliers.
  Alternatives: keep TF32 enabled and relax cache_time tolerances.
  Evidence: TF32-enabled TRT parity shows 5/50 cache_last_time_out failures (max_abs `3.614e-01`); `--noTF32` build passes 50/50 with cache_time max_abs ≤ `1.416e-04` and encoder_output max_abs `1.974e-07`.
  Validation: `artifacts/parity/trt_streaming_parity_cache3_fp32_t57_noTF32_50.json` reports `failed=0`.

- Decision: Pad tail streaming chunks shorter than `T=41` up to the streaming profile target while keeping `length=T_actual`.
  Alternatives: drop tail chunks or broaden TRT profile min T.
  Evidence: Tail chunks <41 previously caused TRT shape errors; after padding + relaxed `encoder_output` time-dim check, untrimmed 1‑utt and 10‑utt runs succeed with 0 empty transcripts.
  Validation: `artifacts/logs/tdt_1utt_trt_fp32_noTF32_untrimmed.*` and `artifacts/logs/tdt_10utt_fp32_noTF32_untrimmed/all_results.json`.

- Decision: Align runtime feature extraction to NeMo `FilterbankFeatures` (preemph=0.97, Slaney mel + Slaney norm, centered STFT with zero padding, window padded to n_fft, log guard 2^-24).
  Alternatives: keep HTK mel + center=false + preemph=0.0 (previous runtime defaults).
  Evidence: `tools/verify_nemo/compare_features.py` now shows tight parity (offline and streaming) with max_abs ≈ `2e-4` and mean_abs ≈ `2e-6` on dev-clean 0000/0002.
  Validation: re-run feature parity spot-checks and ensure WER no longer collapses to punctuation-only outputs.
