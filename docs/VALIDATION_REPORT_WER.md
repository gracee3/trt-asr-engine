# WER Gate Report (LibriSpeech dev gate)

Status: FAIL (transcripts mostly empty; gate blocked).

## Dataset
- Manifest: `eval/manifests/librispeech_dev_gate.tsv`
- Composition: 50 dev-clean + 50 dev-other (pinned)

## Environment
- CLI: `rust/target/debug/cli`
- Model dir: `models/parakeet-tdt-0.6b-v3`
- Runtime env: `LD_LIBRARY_PATH=/home/emmy/git/trt-asr-engine/cpp/build`
- Suite tool: `tools/stt_suite/run_suite.py`

## Runs
### Mode B (streaming-safe)
- Command:
  ```bash
  LD_LIBRARY_PATH=/home/emmy/git/trt-asr-engine/cpp/build \
  PARAKEET_FEATURE_NORM=none \
  python tools/stt_suite/run_suite.py \
    --manifest eval/manifests/librispeech_dev_gate.tsv \
    --cli-path rust/target/debug/cli \
    --model-dir models/parakeet-tdt-0.6b-v3 \
    --variants base \
    --rounds 1 \
    --output-dir /tmp/stt_suite/gate_norm_none
  ```
- WER:
  ```
  base: 100.00% (100 empty)
  ```
- Score output: `/tmp/stt_suite/gate_norm_none/scores.tsv`

### Mode A (model-matching)
- Command:
  ```bash
  LD_LIBRARY_PATH=/home/emmy/git/trt-asr-engine/cpp/build \
  PARAKEET_FEATURE_NORM=per_feature \
  python tools/stt_suite/run_suite.py \
    --manifest eval/manifests/librispeech_dev_gate.tsv \
    --cli-path rust/target/debug/cli \
    --model-dir models/parakeet-tdt-0.6b-v3 \
    --variants base \
    --rounds 1 \
    --output-dir /tmp/stt_suite/gate_norm_per_feature
  ```
- WER:
  ```
  base: 98.23% (52 empty)
  ```
- Score output: `/tmp/stt_suite/gate_norm_per_feature/scores.tsv`

## Blocking Issues
- Transcripts are empty for most utterances under both normalization modes.
- The WER gate cannot be used to lock the normalization decision until decode produces non-empty outputs.

## Next Debug Actions
- Verify TDT decode loop in C++ runtime (blank/duration handling, max symbols per step) against contract.
- Add a short deterministic audio test with expected tokens to detect decode regressions.
- Re-run the gate after decode parity is fixed.
