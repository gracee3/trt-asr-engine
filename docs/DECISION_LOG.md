# Decision Log

## 2026-01-06
- Decision: Use a single JSON contract (`contracts/parakeet-tdt-0.6b-v3.contract.json`) as the canonical runtime contract.
  Alternatives: TOML or YAML.
  Evidence: Requirement in reset plan; JSON matches existing tooling output (`model_meta.json`).
  Validation: Schema check + parity harness uses this contract.

- Decision: Default joint layout is token-first, duration-last.
  Alternatives: duration-first layout via runtime switch.
  Evidence: NeMo TDT loss slices `acts[..., :-n_durations]` vs `acts[..., -n_durations:]` in `nemo/collections/asr/losses/rnnt_pytorch.py`; `model_config.yaml` sets `joint.num_extra_outputs=5` and `loss.tdt_kwargs.durations=[0,1,2,3,4]`; `tools/export_onnx/out/joint.onnx` output is a single linear of size 8198 (no concat).
  Validation: joint ONNX inspected (LogSoftmax axis=-1 output), layout consistent with durations appended at tail; re-export to logits pending, runtime toggle remains for diagnostics.

- Decision: Treat current streaming export as chunk-isolated (`cache_len=0`).
  Alternatives: true stateful cache carryover.
  Evidence: `ONNX_ORT_PARITY_README.md` and `ONNX_PARITY_RESULTS.md` show `cache_last_channel_len_out == 0`.
  Validation: Re-export stateful streaming encoder and re-run closed-loop parity.

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
  Validation: re-export streaming encoder, generate reference JSONL with nonzero cache_len, and pass closed-loop parity with cache feedback.
