# Debugging Guide

This document describes the debugging infrastructure available in trt-asr-engine for diagnosing integration issues, particularly when integrating with audio pipelines like Magnolia.

## Overview

The debugging infrastructure provides:

1. **Audio Taps** - Capture audio at pipeline boundaries for offline analysis
2. **Feature Taps** - Dump mel features being pushed to the engine
3. **NaN/Inf Guards** - Detect and log numeric issues with full context
4. **Cache Length Override** - Debug cache semantics in streaming mode
5. **Replay Harness** - Deterministic reproduction from captured artifacts
6. **Analysis Tools** - Python scripts for waveform/spectrogram visualization

## Quick Reference: Environment Variables

### Audio Taps (for Magnolia integration)

| Variable | Description |
|----------|-------------|
| `AUDIO_TAP_ENABLE=1` | Enable all audio taps globally |
| `AUDIO_TAP_DIR=/path` | Output directory for tap files (default: `.`) |
| `AUDIO_TAP_CAPTURE=1` | Enable capture tap specifically |
| `AUDIO_TAP_POST_DSP=1` | Enable post-DSP tap specifically |
| `AUDIO_TAP_FEATURES=1` | Enable feature tap in trt-asr-engine |

### NaN/Inf Guards

| Variable | Description |
|----------|-------------|
| `PARAKEET_NAN_GUARD_ALWAYS=1` | Check every chunk (default: first 10, then 1-in-100) |
| `PARAKEET_NAN_GUARD_HALT=1` | Abort on first NaN/Inf detection |

### Cache Length Override

| Variable | Description |
|----------|-------------|
| `PARAKEET_CACHE_LEN_OVERRIDE=-1` | Use cache capacity as cache_len_in |
| `PARAKEET_CACHE_LEN_OVERRIDE=N` | Force cache_len_in to specific value N |

### Existing Debug Variables

| Variable | Description |
|----------|-------------|
| `PARAKEET_DEBUG_STAGE_MARKERS=1` | Real-time stage markers to stderr |
| `PARAKEET_DEBUG_DEVICE_SYNC=1` | CUDA device sync timing |
| `PARAKEET_DEBUG_SYNC_MEMCPY=1` | Memory copy timing |
| `PARAKEET_SLOW_MEMCPY_MS=50` | Threshold for slow memcpy warnings |
| `PARAKEET_BLANK_PENALTY=N` | Adjust blank token penalty |
| `PARAKEET_DEBUG_BLANK_SCAN=1` | Log blank-vs-nonblank margin summary per chunk |
| `PARAKEET_Y0_OVERRIDE=N` | Override initial predictor token (skip prompt priming) |
| `PARAKEET_DISABLE_CACHE=1` | Disable encoder cache (for comparison) |

---

## 1. Audio Taps

### Purpose

Audio taps capture raw PCM at pipeline boundaries, allowing offline analysis to isolate whether issues are in:
- Audio capture
- DSP processing (beamforming, noise suppression, resampling)
- Feature extraction
- STT inference

### Tap Points (Recommended)

1. **Capture** - Raw mic input before any DSP
2. **Post-DSP** - After all DSP, before STT feature extraction
3. **Features** - Log-mel features as delivered to the encoder

### Using AudioTapWriter (C++)

Include the header in your Magnolia code:

```cpp
#include "audio_tap.h"  // Copy from trt-asr-engine/cpp/include/

// Create taps at startup
audio_tap::AudioTapWriter capture_tap("capture", 48000, 2,
    audio_tap::Format::S16LE, "raw mic capture pre-DSP");

audio_tap::AudioTapWriter post_dsp_tap("post_dsp", 16000, 1,
    audio_tap::Format::F32LE, "after beamformer and resample");

// In your audio callback:
capture_tap.write_s16(raw_samples, num_samples);

// After DSP processing:
post_dsp_tap.write_f32(processed_samples, num_samples);

// If frames are dropped, record the gap:
post_dsp_tap.record_gap(dropped_sample_count);
```

### Output Files

Each tap produces two files:
- `tap_<name>.raw` - Raw PCM data (append mode)
- `tap_<name>.json` - Metadata sidecar

Example JSON sidecar:
```json
{
  "sample_rate_hz": 16000,
  "channels": 1,
  "format": "f32le",
  "interleaved": true,
  "start_monotonic_ns": 1234567890123,
  "total_samples": 480000,
  "duration_sec": 30.000,
  "stats": {
    "peak": 0.823456,
    "rms": 0.089234,
    "dc_offset": 0.000123,
    "clipped_samples": 0,
    "nan_count": 0,
    "inf_count": 0,
    "gap_samples": 0
  },
  "notes": "after beamformer and resample"
}
```

### Converting to WAV

```bash
# Float32 mono 16kHz
ffmpeg -f f32le -ar 16000 -ac 1 -i tap_post_dsp.raw tap_post_dsp.wav

# Int16 stereo 48kHz
ffmpeg -f s16le -ar 48000 -ac 2 -i tap_capture.raw tap_capture.wav
```

---

## 2. Feature Tap

### Purpose

Dumps the log-mel features being pushed to `parakeet_push_features()`, allowing verification that feature extraction is producing valid data.

### Enabling

```bash
AUDIO_TAP_FEATURES=1 ./your_application
```

### Output

- `tap_features.raw` - f32le, 128 "channels" (mel bins), column-major [C,T]
- `tap_features.json` - Metadata including frame count and stats

### Analyzing Features

```bash
python tools/analyze_tap.py tap_features.raw --features
```

This produces a mel-spectrogram visualization and energy profile plot.

---

## 3. NaN/Inf Guards

### Purpose

Detect numeric issues (NaN, Inf) at critical points with full context for debugging. Particularly useful for diagnosing cache semantics issues that can cause overflow.

### Guard Points

Guards are automatically inserted at:
- **Encoder output** - After encoder inference
- **Cache outputs** - `cache_last_channel_out`, `cache_last_time_out`
- **Joint output** - After joint network inference (FP16 and FP32 paths)

### Alert Format

When NaN/Inf is detected:
```
[parakeet_trt] NAN_GUARD ALERT stage=enc_output nan_count=5 inf_count=0 cache_len_in=256 T_valid=584 chunk_idx=3
[parakeet_trt] NAN_GUARD values[47..57]: -0.123 nan nan 0.456 0.789 -0.234 nan 0.567 0.890 -0.345
```

### Halting on NaN

For debugging, you can make the engine abort on first NaN:

```bash
PARAKEET_NAN_GUARD_HALT=1 ./your_application
```

This allows attaching a debugger or examining the core dump.

---

## 4. Cache Length Override

### Purpose

The streaming encoder uses cache tensors to maintain state across chunks. Invalid `cache_len_in` values can cause out-of-bounds access and NaN propagation.

### Automatic Clamping

The runtime automatically clamps `cache_len_in` to the cache capacity and logs warnings:

```
[parakeet_trt] WARNING: cache_len_in=300 exceeds capacity=256, clamping
```

### Manual Override

For debugging, you can force specific cache_len values:

```bash
# Use full cache capacity
PARAKEET_CACHE_LEN_OVERRIDE=-1 ./your_application

# Force specific value
PARAKEET_CACHE_LEN_OVERRIDE=128 ./your_application
```

### Enhanced Logging

The cache_len logging now shows both clamped and unclamped values:

```
[parakeet_trt] cache_len_in value=256 (unclamped=300) capacity=256 dtype=int64 dims=[1] bytes=8
```

---

## 5. Replay Harness

### Purpose

Deterministic reproduction of issues using captured artifacts. This allows:
- A/B testing with different engine configurations
- Isolating STT issues from audio pipeline issues
- Sharing reproducible test cases

### CLI Usage

The Rust CLI (`rust/cli`) supports multiple replay modes:

#### Replay WAV file (standard)
```bash
./target/debug/cli test.wav --model-dir ./models/parakeet-tdt-0.6b-v3
```

#### Replay raw PCM tap
```bash
./target/debug/cli tap_post_dsp.raw --raw-pcm \
    --model-dir ./models/parakeet-tdt-0.6b-v3 -v
```

#### Replay feature tap (bypass feature extraction)
```bash
./target/debug/cli tap_features.raw --features-input \
    --model-dir ./models/parakeet-tdt-0.6b-v3 -v
```
If `tap_features.json` is present next to the raw file, the CLI will auto-detect
mel bins and layout. You can also pass the JSON sidecar directly:
```bash
./target/debug/cli tap_features.json --features-input \
    --model-dir ./models/parakeet-tdt-0.6b-v3 -v
```

#### Dump features for later replay
```bash
./target/debug/cli test.wav --model-dir ./models/... \
    --dump-features features_dump.raw
```

#### Simulated streaming with verbose output
```bash
./target/debug/cli test.wav --model-dir ./models/... \
    --stream-sim 0.5 -v
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--raw-pcm` | Input is raw PCM (f32le, 16kHz mono) |
| `--sample-rate N` | Sample rate for raw PCM (default: 16000) |
| `--features-input` | Input is pre-computed features (auto-detects tap JSON sidecar) |
| `--n-mels N` | Mel bins for features-input (default: 128, overrides JSON if set) |
| `--dump-features PATH` | Dump computed features to file |
| `--stream-sim SECS` | Simulated streaming interval |
| `-v, --verbose` | Verbose output with timing and stats |

---

## 6. Analysis Tools

### analyze_tap.py

Python script for analyzing tap dumps.

#### Installation

```bash
pip install numpy matplotlib scipy  # Optional: for plots
```

#### Single Tap Analysis

```bash
python tools/analyze_tap.py tap_post_dsp.raw
```

Output:
- Waveform + spectrogram PNG
- Statistics: peak, RMS, DC offset, NaN/Inf counts, clipping
- Warnings for silent audio, phase issues, DC drift

#### Feature Visualization

```bash
python tools/analyze_tap.py tap_features.raw --features
```

Output:
- Mel-spectrogram visualization
- Energy profile over time

#### Compare Multiple Taps

```bash
python tools/analyze_tap.py tap_capture.raw tap_post_dsp.raw --compare
```

Output:
- Side-by-side statistics table
- Energy comparison (dB change between taps)
- Warnings for >20dB energy drops (potential DSP/cancellation issues)

#### Options

| Option | Description |
|--------|-------------|
| `--json PATH` | Specify JSON sidecar (auto-detected if not given) |
| `-o, --output PATH` | Output plot filename |
| `-f, --features` | Treat input as mel-feature dump |
| `--no-plot` | Skip plot generation |
| `-c, --compare` | Compare multiple taps side-by-side |

---

## Debugging Workflow

### Step 1: Establish Baseline

Use the simplest audio path (e.g., SOF direct 16kHz mono) to verify STT works:

```bash
AUDIO_TAP_ENABLE=1 AUDIO_TAP_FEATURES=1 \
    ./magnolia --input=sof_direct
```

### Step 2: Analyze Taps

```bash
python tools/analyze_tap.py tap_*.raw --compare
```

Check for:
- Energy drops between capture and post_dsp (>20dB = problem)
- NaN/Inf in features
- Silent or near-silent RMS

### Step 3: If Audio Looks OK, Check NaN Guards

```bash
PARAKEET_NAN_GUARD_ALWAYS=1 ./your_application
```

Look for `NAN_GUARD ALERT` messages with context.

### Step 4: Try Cache Override

```bash
PARAKEET_CACHE_LEN_OVERRIDE=-1 ./your_application
```

### Step 5: Differential FP32 Test

Build FP32 engines and replay the same features:

```bash
./target/debug/cli tap_features.raw --features-input \
    --model-dir ./models_fp32 -v
```

Compare results. If FP32 works but FP16 doesn't, it's likely numeric sensitivity.

### Step 6: Isolate with Replay

Once you have a failing tap file, you can share it for reproduction:

```bash
# Reproduce issue
./target/debug/cli failing_features.raw --features-input \
    --model-dir ./models/parakeet-tdt-0.6b-v3 -v
```

---

## Interpreting Common Issues

### Issue: RMS ~0 in post_dsp tap

**Likely cause**: Destructive interference in beamformer/downmix

**Check**: Compare `(L+R)` vs `(L-R)` energy in stereo capture tap

### Issue: NaN in encoder output

**Likely cause**: Invalid cache_len causing out-of-bounds indexing

**Check**: Look for `cache_len_in` > `capacity` in logs

### Issue: NaN in joint output only

**Likely cause**: FP16 overflow in joint network

**Check**: Try FP32 joint engine, or check encoder output for extreme values

### Issue: Always blank with cache enabled

**Likely cause**: Cache state not being properly updated/swapped

**Check**: Verify `cache_out_state` logging shows non-zero values after first chunk

### Issue: T_enc always = 1

**Expected**: For streaming mode, T_enc should always be 1 (one timestep per chunk)

**If unexpected**: Check encoder profile configuration

---

## Files Reference

| File | Description |
|------|-------------|
| `cpp/include/audio_tap.h` | C++ RAII audio tap writer |
| `cpp/src/parakeet_trt.cpp` | Main engine with NaN guards and cache override |
| `rust/cli/src/main.rs` | Replay harness CLI |
| `tools/analyze_tap.py` | Python tap analysis tool |
