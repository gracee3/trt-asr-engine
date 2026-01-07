# Contract Sources: parakeet-tdt-0.6b-v3

This document maps every contract field to a source of truth (paper, NeMo config, or export metadata). If a value is derived, the derivation is noted explicitly.

## Source artifacts (primary)
- NeMo model config: `models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo` -> `model_config.yaml`.
- Export metadata: `tools/export_onnx/out/model_meta.json` (also mirrored in `out/model_meta.json`).
- Architecture audit: `audit_model_arch.json` (from `tools/verify_nemo/audit_model_arch.py`).
- Runtime bindings: `docs/runtime_contract.md` (offline encoder/predictor/joint bindings).
- Streaming encoder contract (legacy chunk-isolated): `contracts/encoder_streaming.contract.json`.
- TRT build profiles: `models/parakeet-tdt-0.6b-v3/build_report.json` and `tools/build_trt/README.md`.
- Parity results and streaming mode evidence: `ONNX_ORT_PARITY_README.md`, `ONNX_PARITY_RESULTS.md`.

## Source artifacts (papers)
- TDT greedy inference + duration head: `docs/txt/2304.06795v2.txt` (Algorithm 2, page 5/23).
- Blank + duration=0 disallowed: `docs/txt/2304.06795v2.txt` (page 3/23, note after Eq. 6).
- Max-symbol-per-decoding-step guard: `docs/txt/2304.06795v2.txt` (page 23/23).
- Streaming cache/context + avoid feature normalization: `docs/txt/2312.17279v3.txt` (page 2/8 and 4/8).
- Fast Conformer 8x subsampling + kernel=9 + channels=256: `docs/txt/2305.05084v6.txt` (page 2/8).

## Contract map (field -> source)

### Model identity + hashes
- `model_id`: `tools/export_onnx/out/model_meta.json` -> `model_name`.
- `hashes.nemo_sha256`: `sha256sum models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo`.
- `hashes.model_config_sha256`: SHA256 of `model_config.yaml` extracted from the `.nemo`.
- `hashes.export_meta_sha256`: `sha256sum tools/export_onnx/out/model_meta.json`.

### Features
- `features.sample_rate_hz`: `model_config.yaml` -> `sample_rate`.
- `features.n_fft`: `model_config.yaml` -> `preprocessor.n_fft`.
- `features.n_mels`: `model_config.yaml` -> `preprocessor.features` (also in `model_meta.json` -> `features.n_mels`).
- `features.window_size_sec`: `model_config.yaml` -> `preprocessor.window_size`.
- `features.window_stride_sec`: `model_config.yaml` -> `preprocessor.window_stride`.
- `features.window_length`: derived = `sample_rate * window_size_sec` (16k * 0.025 = 400).
- `features.hop_length`: derived = `sample_rate * window_stride_sec` (16k * 0.01 = 160; matches `model_meta.json`).
- `features.window_type`: `model_config.yaml` -> `preprocessor.window`.
- `features.normalize`: `model_config.yaml` -> `preprocessor.normalize`.
- `features.log_mel`: `model_config.yaml` -> `preprocessor.log`.
- `features.dither`: `model_config.yaml` -> `preprocessor.dither`.

### Tokenizer
- `tokenizer.type`: `model_config.yaml` -> `tokenizer.type` (bpe).
- `tokenizer.model_path`, `tokenizer.vocab_path`: `model_config.yaml` -> `tokenizer.*`.
- `tokenizer.vocab_size`: `model_config.yaml` -> `decoder.vocab_size` (8192) and `vocab.txt` line count.
- `tokenizer.blank_id`: `tools/export_onnx/out/model_meta.json` -> `blank_id` (8192).
- `tokenizer.special_tokens`: `models/parakeet-tdt-0.6b-v3/vocab.txt` (first entries).

### Encoder architecture (offline + streaming)
- `encoder.num_layers`: `model_config.yaml` -> `encoder.n_layers` (24).
- `encoder.d_model`: `model_config.yaml` -> `encoder.d_model` (1024).
- `encoder.n_heads`: `model_config.yaml` -> `encoder.n_heads` (8).
- `encoder.conv_kernel_size`: `model_config.yaml` -> `encoder.conv_kernel_size` (9) and `docs/txt/2305.05084v6.txt` (page 2/8).
- `encoder.subsampling_factor`: `model_config.yaml` -> `encoder.subsampling_factor` (8) and `docs/txt/2305.05084v6.txt` (page 2/8).
- `encoder.subsampling_conv_channels`: `model_config.yaml` -> `encoder.subsampling_conv_channels` (256) and `docs/txt/2305.05084v6.txt` (page 2/8).
- `encoder.att_context_size`, `encoder.att_context_style`: `model_config.yaml` -> `encoder.att_context_*`.

### Encoder IO (offline)
- Inputs/outputs, dtypes, layouts: `docs/runtime_contract.md` and `tools/export_onnx/out/model_meta.json`.
- `encoder_output` shape: `docs/runtime_contract.md` -> `[B, 1024, T_enc]`.

### Encoder IO (streaming + caches)
- Cache input/output shapes: `tools/export_onnx/out/encoder_streaming.onnx` IO summary (batch-first cache layout).
- `valid_out_len=2`: `encoder.streaming_cfg.valid_out_len` after `setup_streaming_params` with clamped cache_drop_size (see `tools/verify_nemo/streaming_encoder_cache.py` run logs).
- Streaming chunk params (`chunk_size`, `shift_size`, `cache_drop_size`, `pre_encode_cache_size`, `drop_extra_pre_encoded`): `encoder.streaming_cfg` after `setup_streaming_params` (see `tools/verify_nemo/streaming_encoder_cache.py` run logs).
- `cache_drop_size` clamp (72 → 71): derived by measuring pre-encode length for chunk0 (577/584 frames) minus `drop_extra_pre_encoded=2` to avoid negative cache_len (see `tools/verify_nemo/streaming_encoder_cache.py` output + NeMo pre_encode behavior).
- Cache semantics + need for context-limited caching: `docs/txt/2312.17279v3.txt` (page 4/8).
- Stateful cache carryover behavior: implemented in NeMo `ConformerEncoder.forward_internal` and `streaming_post_process` (`nemo/collections/asr/modules/conformer_encoder.py`); validated via ORT closed-loop parity on `encoder_streaming.onnx`.

### Predictor
- `pred_hidden`: `model_config.yaml` -> `model_defaults.pred_hidden` (640) and `decoder.prednet.pred_hidden`.
- `pred_rnn_layers`: `model_config.yaml` -> `decoder.prednet.pred_rnn_layers` (2).
- IO shapes and dtypes: `docs/runtime_contract.md`.

### Joint
- `joint_vocab_size`: `tools/export_onnx/out/model_meta.json` -> `joint_vocab_size` (8198).
- `num_classes` (token vocab): `model_config.yaml` -> `joint.num_classes` (8192).
- `duration_values`: `model_config.yaml` -> `model_defaults.tdt_durations` and `tools/export_onnx/out/model_meta.json`.
- IO shapes: `docs/runtime_contract.md` and `tools/export_onnx/README.md`.
- `joint_output` normalization target: raw logits (export forces `joint.log_softmax=False` in `tools/export_onnx/export.py`; verify after re-export).

### Decode invariants (TDT)
- Greedy decode algorithm (token argmax + duration argmax, advance time by duration): `docs/txt/2304.06795v2.txt` (Algorithm 2, page 5/23).
- Blank + duration=0 disallowed (requires renorm or override): `docs/txt/2304.06795v2.txt` (page 3/23 note after Eq. 6).
- Max symbols per decoding step guard (avoid infinite loops): `docs/txt/2304.06795v2.txt` (page 23/23).
- Joiner head ordering (token+blank first, durations last): evidence from `nemo/collections/asr/losses/rnnt_pytorch.py` (splits `acts[..., :-n_durations]` vs `acts[..., -n_durations:]`), `model_config.yaml` (`joint.num_extra_outputs=5`, `loss.tdt_kwargs.durations=[0,1,2,3,4]`), and `tools/export_onnx/out/joint.onnx` (single linear output size 8198, no concat).

### TRT profiles + tolerances
- Offline component profiles: `models/parakeet-tdt-0.6b-v3/build_report.json` and `tools/build_trt/README.md`.
- Streaming encoder profiles: `contracts/encoder_streaming.contract.json`.
- ORT tolerance baseline: `ONNX_PARITY_RESULTS.md` (atol=1e-4, rtol=1e-4; cache_last_time_out requires relaxed tolerance).
- TRT acceptance guidance: `TRT_INTEGRATION_CHECKLIST.md` and `ONNX_ORT_PARITY_README.md`.

## Provenance gaps to close
- Resolve cache size mismatch: `last_channel_cache_size=10000` (NeMo config) vs `cache_size=256` (streaming export).
- Decide blank + duration=0 handling policy for decode (paper allows renorm; runtime currently uses a heuristic).
- Confirm predictor cell type (LSTM vs other) from NeMo modules.
- Validate feature normalization policy for streaming equivalence (per-feature vs no normalization).
