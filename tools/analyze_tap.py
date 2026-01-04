#!/usr/bin/env python3
"""
analyze_tap.py - Analyze audio tap dumps from trt-asr-engine or Magnolia

Reads raw PCM + JSON sidecar files and produces:
- Waveform plot
- Spectrogram plot
- Statistics (RMS, peak, DC offset, clipping, NaN/Inf counts)
- Optional: channel correlation for stereo

Usage:
    python analyze_tap.py tap_post_dsp.raw
    python analyze_tap.py tap_features.raw --features  # For mel-feature dumps
    python analyze_tap.py tap_capture.raw --output report.png

Environment variables:
    AUDIO_TAP_DIR - Directory where tap files are located (default: current dir)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Optional dependencies for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Optional: scipy for spectrogram
try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_raw_audio(raw_path: str, json_path: Optional[str] = None) -> Tuple[np.ndarray, dict]:
    """Load raw PCM file with optional JSON sidecar metadata."""
    raw_path = Path(raw_path)

    # Try to find JSON sidecar
    if json_path is None:
        json_path = raw_path.with_suffix('.json')
    else:
        json_path = Path(json_path)

    # Default metadata
    meta = {
        'sample_rate_hz': 16000,
        'channels': 1,
        'format': 'f32le',
        'interleaved': True,
        'notes': '',
    }

    if json_path.exists():
        with open(json_path, 'r') as f:
            meta.update(json.load(f))
        print(f"Loaded metadata from {json_path}")
    else:
        print(f"Warning: No JSON sidecar found at {json_path}, using defaults")

    # Determine numpy dtype from format
    fmt_to_dtype = {
        's16le': np.int16,
        's32le': np.int32,
        'f32le': np.float32,
        'f64le': np.float64,
    }

    fmt = meta.get('format', 'f32le').lower()
    if fmt not in fmt_to_dtype:
        print(f"Warning: Unknown format '{fmt}', assuming f32le")
        fmt = 'f32le'

    dtype = fmt_to_dtype[fmt]

    # Load raw data
    data = np.fromfile(raw_path, dtype=dtype)

    # Reshape for multi-channel
    channels = meta.get('channels', 1)
    if channels > 1 and meta.get('interleaved', True):
        # Interleaved: reshape to [frames, channels]
        num_frames = len(data) // channels
        data = data[:num_frames * channels].reshape(num_frames, channels)

    return data, meta


def compute_stats(data: np.ndarray, meta: dict) -> dict:
    """Compute audio statistics."""
    # Flatten for stats if multi-channel
    flat = data.flatten().astype(np.float64)

    # Handle NaN/Inf
    nan_mask = np.isnan(flat)
    inf_mask = np.isinf(flat)
    finite_mask = np.isfinite(flat)
    finite_data = flat[finite_mask]

    stats = {
        'total_samples': len(flat),
        'nan_count': int(np.sum(nan_mask)),
        'inf_count': int(np.sum(inf_mask)),
        'finite_count': int(np.sum(finite_mask)),
    }

    if len(finite_data) > 0:
        stats['peak'] = float(np.max(np.abs(finite_data)))
        stats['rms'] = float(np.sqrt(np.mean(finite_data ** 2)))
        stats['dc_offset'] = float(np.mean(finite_data))
        stats['min'] = float(np.min(finite_data))
        stats['max'] = float(np.max(finite_data))
        stats['std'] = float(np.std(finite_data))
    else:
        stats['peak'] = stats['rms'] = stats['dc_offset'] = 0.0
        stats['min'] = stats['max'] = stats['std'] = 0.0

    # Clipping detection for integer formats
    fmt = meta.get('format', 'f32le').lower()
    if fmt == 's16le':
        clipped = np.sum((data == np.iinfo(np.int16).max) | (data == np.iinfo(np.int16).min))
        stats['clipped_samples'] = int(clipped)
    elif fmt == 's32le':
        clipped = np.sum((data == np.iinfo(np.int32).max) | (data == np.iinfo(np.int32).min))
        stats['clipped_samples'] = int(clipped)
    else:
        stats['clipped_samples'] = 0

    # Duration
    sr = meta.get('sample_rate_hz', 16000)
    channels = meta.get('channels', 1)
    num_frames = stats['total_samples'] // channels
    stats['duration_sec'] = num_frames / sr

    return stats


def compute_channel_stats(data: np.ndarray) -> dict:
    """Compute stereo channel statistics."""
    if data.ndim != 2 or data.shape[1] < 2:
        return {}

    L = data[:, 0].astype(np.float64)
    R = data[:, 1].astype(np.float64)

    # Correlation
    if np.std(L) > 0 and np.std(R) > 0:
        corr = float(np.corrcoef(L, R)[0, 1])
    else:
        corr = 0.0

    # L+R vs L-R energy (detects phase issues)
    sum_energy = np.sum((L + R) ** 2)
    diff_energy = np.sum((L - R) ** 2)

    return {
        'channel_correlation': corr,
        'sum_energy': float(sum_energy),
        'diff_energy': float(diff_energy),
        'sum_diff_ratio': float(sum_energy / diff_energy) if diff_energy > 0 else float('inf'),
    }


def plot_audio(data: np.ndarray, meta: dict, output_path: str, is_features: bool = False):
    """Generate waveform and spectrogram plots."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
        return

    sr = meta.get('sample_rate_hz', 16000)
    channels = meta.get('channels', 1)

    if is_features:
        # Feature dump: data is [T, 128] mel features
        plot_features(data, meta, output_path)
        return

    # Regular audio
    if data.ndim == 2:
        # Multi-channel: use first channel for plotting
        audio = data[:, 0].astype(np.float32)
    else:
        audio = data.astype(np.float32)

    # Normalize for plotting
    if meta.get('format', 'f32le').startswith('s'):
        # Integer format: normalize to [-1, 1]
        dtype_info = np.iinfo(np.int16) if 's16' in meta.get('format', '') else np.iinfo(np.int32)
        audio = audio / dtype_info.max

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Waveform
    t = np.arange(len(audio)) / sr
    axes[0].plot(t, audio, linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f"Waveform - {meta.get('notes', 'audio tap')}")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, t[-1] if len(t) > 0 else 1)

    # Spectrogram
    if HAS_SCIPY and len(audio) > 256:
        nperseg = min(512, len(audio) // 4)
        noverlap = nperseg // 2
        f, t_spec, Sxx = signal.spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)

        # Log scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        im = axes[1].pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap='magma')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Spectrogram')
        plt.colorbar(im, ax=axes[1], label='Power (dB)')
    else:
        axes[1].text(0.5, 0.5, 'Spectrogram unavailable\n(scipy not installed or audio too short)',
                     ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_features(data: np.ndarray, meta: dict, output_path: str):
    """Plot mel-spectrogram features."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
        return

    channels = meta.get('channels', 128)  # Mel bins
    frame_rate = meta.get('sample_rate_hz', 100)  # Frame rate (100 Hz = 10ms)

    # Reshape to [T, mel_bins] if needed
    if data.ndim == 1:
        num_frames = len(data) // channels
        data = data[:num_frames * channels].reshape(num_frames, channels)

    # Transpose to [mel_bins, T] for imshow
    mel_spec = data.T

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Mel spectrogram
    t = np.arange(mel_spec.shape[1]) / frame_rate
    mel_bins = np.arange(mel_spec.shape[0])

    im = axes[0].imshow(mel_spec, aspect='auto', origin='lower',
                         extent=[0, t[-1] if len(t) > 0 else 1, 0, channels],
                         cmap='viridis')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Mel bin')
    axes[0].set_title(f"Log-Mel Features - {meta.get('notes', 'features tap')}")
    plt.colorbar(im, ax=axes[0], label='Log energy')

    # Energy over time (sum across mel bins)
    energy = np.sum(mel_spec, axis=0)
    axes[1].plot(t, energy, linewidth=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Sum of log-mel energy')
    axes[1].set_title('Energy profile over time')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, t[-1] if len(t) > 0 else 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved features plot to {output_path}")


def print_stats(stats: dict, channel_stats: dict, meta: dict):
    """Print statistics in a readable format."""
    print("\n" + "=" * 60)
    print("AUDIO TAP ANALYSIS")
    print("=" * 60)

    print(f"\nMetadata:")
    print(f"  Sample rate:  {meta.get('sample_rate_hz', 'unknown')} Hz")
    print(f"  Channels:     {meta.get('channels', 'unknown')}")
    print(f"  Format:       {meta.get('format', 'unknown')}")
    print(f"  Notes:        {meta.get('notes', '')}")

    print(f"\nDuration:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Duration:      {stats['duration_sec']:.3f} sec")

    print(f"\nAmplitude statistics:")
    print(f"  Peak:      {stats['peak']:.6f}")
    print(f"  RMS:       {stats['rms']:.6f}")
    print(f"  DC offset: {stats['dc_offset']:.6f}")
    print(f"  Min:       {stats['min']:.6f}")
    print(f"  Max:       {stats['max']:.6f}")
    print(f"  Std dev:   {stats['std']:.6f}")

    print(f"\nData quality:")
    print(f"  NaN samples:     {stats['nan_count']:,}")
    print(f"  Inf samples:     {stats['inf_count']:,}")
    print(f"  Finite samples:  {stats['finite_count']:,}")
    print(f"  Clipped samples: {stats['clipped_samples']:,}")

    if channel_stats:
        print(f"\nStereo analysis:")
        print(f"  Channel correlation: {channel_stats['channel_correlation']:.4f}")
        print(f"  (L+R) energy:        {channel_stats['sum_energy']:.2e}")
        print(f"  (L-R) energy:        {channel_stats['diff_energy']:.2e}")
        print(f"  Sum/Diff ratio:      {channel_stats['sum_diff_ratio']:.2f}")

        if channel_stats['sum_diff_ratio'] < 2.0:
            print("  WARNING: Low sum/diff ratio may indicate phase cancellation!")

    # Warnings
    print("\nDiagnostics:")
    if stats['nan_count'] > 0:
        print(f"  WARNING: {stats['nan_count']} NaN values detected!")
    if stats['inf_count'] > 0:
        print(f"  WARNING: {stats['inf_count']} Inf values detected!")
    if stats['clipped_samples'] > 0:
        clip_pct = 100 * stats['clipped_samples'] / stats['total_samples']
        print(f"  WARNING: {clip_pct:.2f}% of samples are clipped!")
    if abs(stats['dc_offset']) > 0.01 and stats['rms'] > 0:
        dc_ratio = abs(stats['dc_offset']) / stats['rms']
        if dc_ratio > 0.1:
            print(f"  WARNING: Significant DC offset (DC/RMS = {dc_ratio:.2f})")
    if stats['rms'] < 1e-6:
        print("  WARNING: RMS is near zero - audio may be silent or near-silent!")
    if stats['peak'] < 1e-4:
        print("  WARNING: Peak is very low - audio may be effectively silent!")

    print("=" * 60 + "\n")


def compare_taps(tap_files: list):
    """Compare multiple tap files side-by-side."""
    print("\n" + "=" * 60)
    print("TAP COMPARISON")
    print("=" * 60)

    results = []
    for f in tap_files:
        try:
            data, meta = load_raw_audio(f)
            stats = compute_stats(data, meta)
            results.append((f, meta, stats))
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not results:
        return

    # Print comparison table
    headers = ['Tap', 'SR', 'Ch', 'Duration', 'RMS', 'Peak', 'DC', 'NaN', 'Inf']
    print(f"\n{headers[0]:<30} {headers[1]:>8} {headers[2]:>4} {headers[3]:>10} {headers[4]:>10} {headers[5]:>10} {headers[6]:>10} {headers[7]:>6} {headers[8]:>6}")
    print("-" * 106)

    for path, meta, stats in results:
        name = Path(path).stem[:28]
        print(f"{name:<30} {meta.get('sample_rate_hz', 0):>8} {meta.get('channels', 1):>4} "
              f"{stats['duration_sec']:>10.3f} {stats['rms']:>10.6f} {stats['peak']:>10.6f} "
              f"{stats['dc_offset']:>10.6f} {stats['nan_count']:>6} {stats['inf_count']:>6}")

    # Check for energy drop between taps
    if len(results) >= 2:
        print("\nEnergy comparison:")
        for i in range(1, len(results)):
            prev_rms = results[i-1][2]['rms']
            curr_rms = results[i][2]['rms']
            if prev_rms > 0:
                ratio = curr_rms / prev_rms
                db_change = 20 * np.log10(ratio) if ratio > 0 else float('-inf')
                print(f"  {Path(results[i-1][0]).stem} -> {Path(results[i][0]).stem}: {db_change:+.1f} dB")
                if ratio < 0.1:
                    print(f"    WARNING: >20dB drop may indicate DSP/cancellation issue!")


def main():
    parser = argparse.ArgumentParser(description='Analyze audio tap dumps')
    parser.add_argument('files', nargs='+', help='Raw PCM file(s) to analyze')
    parser.add_argument('--json', help='JSON sidecar file (auto-detected if not specified)')
    parser.add_argument('--output', '-o', help='Output plot filename (default: <input>_analysis.png)')
    parser.add_argument('--features', '-f', action='store_true',
                        help='Treat input as mel-feature dump (128 bins)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare multiple taps side-by-side')

    args = parser.parse_args()

    if args.compare and len(args.files) > 1:
        compare_taps(args.files)
        return

    for input_file in args.files:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {input_file}")
        print(f"{'=' * 60}")

        try:
            data, meta = load_raw_audio(input_file, args.json)
        except Exception as e:
            print(f"Error loading {input_file}: {e}")
            continue

        # Compute stats
        stats = compute_stats(data, meta)
        channel_stats = compute_channel_stats(data) if data.ndim == 2 and data.shape[1] >= 2 else {}

        # Print stats
        print_stats(stats, channel_stats, meta)

        # Generate plot
        if not args.no_plot:
            output_path = args.output or str(Path(input_file).with_suffix('')) + '_analysis.png'
            try:
                plot_audio(data, meta, output_path, is_features=args.features)
            except Exception as e:
                print(f"Error generating plot: {e}")


if __name__ == '__main__':
    main()
