# AGENTS.md - Deterministic STT Debug Suite (No Microphone)

This runbook defines a fully automated, deterministic test pipeline for:
1. trt-asr-engine standalone decoding (CLI + feature replay)
2. Magnolia end-to-end (Magnolia -> DSP -> Parakeet STT), using a virtual mic (ALSA loopback)
3. Parity/instrumentation checks available in trt-asr-engine/tools
4. LibriSpeech-based regression with transcript scoring

## Why This Exists

Magnolia is still evolving; we must not assume the live audio path is correct. All test inputs come from known audio files (LibriSpeech), and the capture path uses an isolated loopback device (no interaction with system audio).

## Quick Start

```bash
# 1. Setup ALSA loopback (one-time per boot, optional but recommended for Magnolia testing)
sudo modprobe snd-aloop index=10 id=LoopSTT

# 2. Build manifest from LibriSpeech (converts FLAC -> 16kHz mono WAV)
python tools/stt_suite/make_manifest.py \
    ~/git/magnolia/tools/LibriSpeech/dev-clean \
    --output /tmp/stt_suite/manifest.tsv \
    --wav-dir /tmp/stt_suite/wav \
    --num-utterances 25 \
    --verify

# 3. Run the test suite
python tools/stt_suite/run_suite.py \
    --manifest /tmp/stt_suite/manifest.tsv \
    --cli-path ./rust/target/debug/cli \
    --model-dir ./models/parakeet \
    --rounds 2 \
    --verbose

# 4. Score results
python tools/stt_suite/score_wer.py \
    /tmp/stt_suite/suite_<timestamp>_<pid>/ \
    --manifest /tmp/stt_suite/manifest.tsv
```

## Canonical Environment + Invariants

- Use ALSA loopback card `LoopSTT` (no system audio disturbance)
- Use deterministic Parakeet runtime flags:
  - `PARAKEET_MAX_FRAMES_PER_PUSH=256` (streaming chunking; mitigates cache overflow / NaNs)
  - Token-first joiner layout (duration logits last). (Matches TDT reference: tokens then durations)
- Always run with taps + summary TSV
- Any code change must be validated by running the suite matrix (base/nopunct/nocache/nocache_nopunct)

## Test Variants

| Variant | Environment Variables | Purpose |
|---------|----------------------|---------|
| `base` | (none) | Default behavior |
| `nopunct` | `PARAKEET_DISABLE_PUNCT_SUPPRESSION=1` | Disable punctuation suppression |
| `nocache` | `PARAKEET_DISABLE_CACHE=1` | Disable encoder cache |
| `nocache_nopunct` | Both flags | Baseline without cache or punct |

Common environment (always set):
- `PARAKEET_MAX_FRAMES_PER_PUSH=256`
- `AUDIO_TAP_ENABLE=1`
- `AUDIO_TAP_DIR=<run_dir>/taps`
- `AUDIO_TAP_FEATURES=1`

## ALSA Loopback Setup (Virtual Microphone)

The ALSA loopback creates a dedicated audio path that doesn't interfere with system audio.

### Load Loopback Module

```bash
# One-time per boot
sudo modprobe snd-aloop index=10 id=LoopSTT

# Verify
aplay -l | grep -i loop
arecord -l | grep -i loop
```

### Device Endpoints

- **Playback (write audio)**: `hw:LoopSTT,0,0` or `plughw:LoopSTT,0,0`
- **Capture (read audio)**: `hw:LoopSTT,1,0` or `plughw:LoopSTT,1,0`

### Optional: Stable Device Names (~/.asoundrc)

```ini
# ~/.asoundrc - LoopSTT virtual mic / playback
pcm.loopstt_play {
  type plug
  slave {
    pcm "hw:LoopSTT,0,0"
  }
}

pcm.loopstt_cap_mono16k {
  type plug
  slave {
    pcm "hw:LoopSTT,1,0"
    rate 16000
    channels 1
    format S16_LE
  }
}
```

### Sanity Check

```bash
# Play into loopback while capturing
aplay -D plughw:LoopSTT,0,0 -q test_16k_mono.wav &
arecord -D plughw:LoopSTT,1,0 -r 16000 -c 1 -f S16_LE -d 3 /tmp/loop_cap.wav

# Verify captured audio
ffprobe /tmp/loop_cap.wav
```

## Dataset: LibriSpeech

### Structure

```
~/git/magnolia/tools/LibriSpeech/
├── dev-clean/
│   └── <speaker_id>/
│       └── <chapter_id>/
│           ├── <speaker>-<chapter>.trans.txt
│           └── <speaker>-<chapter>-<utt>.flac
└── test-clean/
    └── ...
```

### Convert FLAC to WAV

Using ffmpeg:
```bash
ffmpeg -y -i input.flac -ar 16000 -ac 1 -c:a pcm_s16le out.wav
```

Using sox:
```bash
sox input.flac -r 16000 -c 1 -b 16 -e signed-integer out.wav
```

### Build Manifest

```bash
python tools/stt_suite/make_manifest.py \
    ~/git/magnolia/tools/LibriSpeech/dev-clean \
    --output manifest.tsv \
    --wav-dir ./wav \
    --num-utterances 25 \
    --verify \
    --verbose
```

Output format (`manifest.tsv`):
```
utt_id	wav_path	ref_text
1221-135766-0000	/path/to/1221-135766-0000.wav	HOW STRANGE IT SEEMED...
```

## Running the Suite

### Full Matrix

```bash
python tools/stt_suite/run_suite.py \
    --manifest manifest.tsv \
    --cli-path ./rust/target/debug/cli \
    --model-dir ./models/parakeet \
    --rounds 2 \
    --verbose
```

### Single Variant

```bash
python tools/stt_suite/run_suite.py \
    --manifest manifest.tsv \
    --cli-path ./rust/target/debug/cli \
    --model-dir ./models/parakeet \
    --variants base \
    --rounds 1
```

### With Loopback (Magnolia Integration)

```bash
python tools/stt_suite/run_suite.py \
    --manifest manifest.tsv \
    --cli-path ./rust/target/debug/cli \
    --model-dir ./models/parakeet \
    --use-loopback \
    --test-loopback  # Just test loopback setup, then exit
```

## Output Structure

```
/tmp/stt_suite/suite_<timestamp>_<pid>/
├── all_results.json
├── loopback_test/
│   └── loopback_test.wav
├── base/
│   └── round_0/
│       ├── transcripts.tsv
│       ├── summary.json
│       ├── taps/
│       │   ├── *.raw
│       │   └── *.json
│       └── <utt_id>_debug.log
├── nopunct/
│   └── ...
├── nocache/
│   └── ...
└── nocache_nopunct/
    └── ...
```

## Scoring

```bash
python tools/stt_suite/score_wer.py \
    /tmp/stt_suite/suite_<timestamp>_<pid>/ \
    --manifest manifest.tsv \
    --verbose
```

Output:
- `scores.tsv`: Per-utterance WER scores
- Summary table to stdout

## Debugging Failures

### Empty Transcripts

1. Check if punctuation suppression is the cause:
   ```bash
   # Compare base vs nopunct
   diff <(cut -f2 base/round_0/transcripts.tsv) \
        <(cut -f2 nopunct/round_0/transcripts.tsv)
   ```

2. Check debug logs for errors:
   ```bash
   grep -i error */round_*/*_debug.log
   ```

### NaN Detection

1. Check debug output for NaN counts:
   ```bash
   grep -i nan */round_*/*_debug.log
   ```

2. If NaNs detected, check if `PARAKEET_MAX_FRAMES_PER_PUSH=256` is set

### Cache Issues

1. Compare nocache vs base variants
2. Use `diagnose_cache_time_mismatch.py`:
   ```bash
   python tools/onnxruntime/diagnose_cache_time_mismatch.py \
       --wav-path <wav_file> \
       --model-dir ./models/parakeet
   ```

## Parity Tools Integration

These existing tools can be run on the same manifest subset:

```bash
# ONNX streaming parity
python tools/onnxruntime/onnx_streaming_parity.py \
    --wav-path <wav_file> \
    --model-dir ./models/parakeet

# TensorRT streaming parity
python tools/tensorrt/trt_streaming_parity.py \
    --wav-path <wav_file> \
    --model-dir ./models/parakeet

# Cache time mismatch diagnosis
python tools/onnxruntime/diagnose_cache_time_mismatch.py \
    --wav-path <wav_file> \
    --model-dir ./models/parakeet

# NeMo reference verification
python tools/verify_nemo/verify.py \
    --wav-path <wav_file> \
    --nemo-model <nemo_checkpoint>

# Inspect TRT engine bindings
python tools/build_trt/scripts/inspect_engine.py \
    ./models/parakeet/encoder.trt
```

## Acceptance Criteria

A change is acceptable only if:

1. **No NaNs** in encoder/joint across the full suite matrix
2. **Non-empty text** for at least the `nocache_nopunct` condition
3. **Punct-only emission** does not dominate (or is explainable)
4. **WER stable** or improved vs baseline across repeated runs

## Paper Alignment

### TDT (Token-and-Duration Transducer)

The joiner output is conceptually `[token_logits..., duration_logits...]` with independent normalization:
- First `vocab_size` outputs are token logits (softmax over vocabulary)
- Remaining outputs are duration logits (softmax over duration values)
- Token-first ordering is the expected layout

Reference: *TDT: Token-and-Duration Transducer for Speech Recognition* (2023)

### Fast Conformer

- Aggressive downsampling reduces encoder compute
- Streaming stability depends on sensible chunking boundaries
- `PARAKEET_MAX_FRAMES_PER_PUSH=256` maintains consistent feature rate

Reference: *Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition* (2023)

## Repo Hygiene

### Clean Testing from Known-Good State

```bash
cd ~/git/trt-asr-engine
git fetch origin
git status

# Option 1: Worktree (recommended)
git worktree add ../trt-asr-engine-wt/stt_suite origin/main
cd ../trt-asr-engine-wt/stt_suite

# Option 2: Stash and checkout
git stash push -u -m "wip before stt suite"
git checkout -B stt_suite origin/main
```

### Build

```bash
# C++ runtime
cmake --build cpp/build -j

# Rust CLI
cargo build -p cli
```

## Troubleshooting

### "aplay: main:831: audio open error: No such file or directory"

The loopback device is not loaded:
```bash
sudo modprobe snd-aloop index=10 id=LoopSTT
```

### "Model files not found"

Check model directory structure:
```bash
ls -la ./models/parakeet/
# Should contain: encoder.trt (or .plan/.engine), joint.trt, predictor.trt
```

### "WAV must be 16kHz mono"

Re-run manifest creation with conversion:
```bash
python tools/stt_suite/make_manifest.py \
    /path/to/librispeech \
    --output manifest.tsv \
    --wav-dir ./wav \
    --verify
```

### "Command timed out"

Increase timeout or check for infinite loops:
```bash
python tools/stt_suite/run_suite.py \
    --manifest manifest.tsv \
    --cli-path ./rust/target/debug/cli \
    --model-dir ./models/parakeet \
    --num-utterances 1 \
    --verbose
```
