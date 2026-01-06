# Decision Log

## 2026-01-06
- Decision: Use a single JSON contract (`contracts/parakeet-tdt-0.6b-v3.contract.json`) as the canonical runtime contract.
  Alternatives: TOML or YAML.
  Evidence: Requirement in reset plan; JSON matches existing tooling output (`model_meta.json`).
  Validation: Schema check + parity harness uses this contract.

- Decision: Default joint layout is token-first, duration-last.
  Alternatives: duration-first layout via runtime switch.
  Evidence: `tools/stt_suite/AGENTS.md` and `PARAKEET_JOINT_DUR_FIRST` runtime toggle.
  Validation: Inspect `joint.onnx` output ordering and add a unit test that slices logits.

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
