#!/usr/bin/env python3
"""
analyze_tap.py - Analyze audio tap dumps from trt-asr-engine or Magnolia

Reads raw PCM + JSON sidecar files and produces:
- Waveform plot
- Spectrogram plot (or mel-spectrogram for features)
- Statistics (RMS, peak, DC offset in both linear and dBFS)
- Optional: channel correlation for stereo

Usage:
    python analyze_tap.py tap_post_dsp.raw
    python analyze_tap.py tap_features.raw --features
    python analyze_tap.py tap_capture.raw tap_post_dsp.raw --compare

The script automatically detects JSON sidecars and uses metadata for proper
normalization and interpretation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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


# ============================================================================
# Fullscale values for different formats (for dBFS calculation)
# ============================================================================

FORMAT_FULLSCALE = {
    's16le': 32768.0,
    's32le': 2147483648.0,
    'f32le': 1.0,
    'f64le': 1.0,
}


def linear_to_dbfs(value: float, fullscale: float) -> float:
    """Convert linear amplitude to dBFS."""
    if value <= 0 or fullscale <= 0:
        return -120.0  # Floor
    return 20.0 * np.log10(value / fullscale)


def load_raw_audio(raw_path: str, json_path: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        'layout': 'interleaved_frames',
        'fullscale': 1.0,
        'notes': '',
        'kind': 'audio',  # or 'mel_features'
    }

    if json_path.exists():
        with open(json_path, 'r') as f:
            loaded = json.load(f)
            meta.update(loaded)
        print(f"Loaded metadata from {json_path}")

        # Extract fullscale from metadata if present
        if 'fullscale' not in loaded:
            meta['fullscale'] = FORMAT_FULLSCALE.get(meta.get('format', 'f32le'), 1.0)
    else:
        print(f"Warning: No JSON sidecar found at {json_path}, using defaults")
        meta['fullscale'] = FORMAT_FULLSCALE.get(meta.get('format', 'f32le'), 1.0)

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
    if channels > 1 and meta.get('layout', 'interleaved_frames') == 'interleaved_frames':
        # Interleaved: reshape to [frames, channels]
        num_frames = len(data) // channels
        data = data[:num_frames * channels].reshape(num_frames, channels)

    return data, meta


def compute_stats(data: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Compute audio statistics with dBFS normalization."""
    fullscale = meta.get('fullscale', 1.0)

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
        'fullscale': fullscale,
    }

    if len(finite_data) > 0:
        stats['peak'] = float(np.max(np.abs(finite_data)))
        stats['rms'] = float(np.sqrt(np.mean(finite_data ** 2)))
        stats['dc_offset'] = float(np.mean(finite_data))
        stats['min'] = float(np.min(finite_data))
        stats['max'] = float(np.max(finite_data))
        stats['std'] = float(np.std(finite_data))

        # dBFS values
        stats['peak_dbfs'] = linear_to_dbfs(stats['peak'], fullscale)
        stats['rms_dbfs'] = linear_to_dbfs(stats['rms'], fullscale)
    else:
        stats['peak'] = stats['rms'] = stats['dc_offset'] = 0.0
        stats['min'] = stats['max'] = stats['std'] = 0.0
        stats['peak_dbfs'] = stats['rms_dbfs'] = -120.0

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
        # Check for potential missing normalization in float data
        if fmt in ('f32le', 'f64le') and stats['peak'] > 2.0:
            stats['scaling_warning'] = f"Peak {stats['peak']:.2f} > 2.0 suggests missing normalization"

    # Duration
    sr = meta.get('sample_rate_hz', 16000)
    channels = meta.get('channels', 1)
    num_frames = stats['total_samples'] // channels
    stats['duration_sec'] = num_frames / sr

    return stats


def compute_channel_stats(data: np.ndarray) -> Dict[str, Any]:
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
    sum_signal = L + R
    diff_signal = L - R
    sum_energy = float(np.sum(sum_signal ** 2))
    diff_energy = float(np.sum(diff_signal ** 2))

    return {
        'channel_correlation': corr,
        'sum_energy': sum_energy,
        'diff_energy': diff_energy,
        'sum_diff_ratio': sum_energy / diff_energy if diff_energy > 0 else float('inf'),
        'sum_rms': float(np.sqrt(np.mean(sum_signal ** 2))),
        'diff_rms': float(np.sqrt(np.mean(diff_signal ** 2))),
    }


def plot_audio(data: np.ndarray, meta: Dict[str, Any], output_path: str, is_features: bool = False):
    """Generate waveform and spectrogram plots."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
        return

    sr = meta.get('sample_rate_hz', 16000)

    if is_features or meta.get('kind') == 'mel_features':
        plot_features(data, meta, output_path)
        return

    # Regular audio
    if data.ndim == 2:
        audio = data[:, 0].astype(np.float32)
    else:
        audio = data.astype(np.float32)

    # Normalize to [-1, 1] for plotting
    fullscale = meta.get('fullscale', 1.0)
    if fullscale != 1.0:
        audio = audio / fullscale

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Waveform
    t = np.arange(len(audio)) / sr
    axes[0].plot(t, audio, linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (normalized)')
    axes[0].set_title(f"Waveform - {meta.get('tap_name', meta.get('notes', 'audio tap'))}")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, t[-1] if len(t) > 0 else 1)
    axes[0].set_ylim(-1.1, 1.1)

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


def plot_features(data: np.ndarray, meta: Dict[str, Any], output_path: str):
    """Plot mel-spectrogram features."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots")
        return

    mel_bins = meta.get('mel_bins', meta.get('channels', 128))
    frame_rate = meta.get('frame_rate_hz', 1000.0 / meta.get('frame_shift_ms', 10.0))
    layout = meta.get('layout', 'bins_major')

    # Reshape to [T, mel_bins] if needed
    if data.ndim == 1:
        num_frames = len(data) // mel_bins
        data = data[:num_frames * mel_bins].reshape(num_frames, mel_bins)
    elif layout == 'bins_major' and data.ndim == 2:
        # [C, T] -> [T, C]
        data = data.T

    # Now data is [T, mel_bins], transpose for imshow
    mel_spec = data.T  # [mel_bins, T]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Mel spectrogram
    t = np.arange(mel_spec.shape[1]) / frame_rate

    im = axes[0].imshow(mel_spec, aspect='auto', origin='lower',
                         extent=[0, t[-1] if len(t) > 0 else 1, 0, mel_bins],
                         cmap='viridis')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Mel bin')
    axes[0].set_title(f"Log-Mel Features - {meta.get('tap_name', meta.get('notes', 'features tap'))}")
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


def print_stats(stats: Dict[str, Any], channel_stats: Dict[str, Any], meta: Dict[str, Any]):
    """Print statistics in a readable format."""
    print("\n" + "=" * 60)
    print("AUDIO TAP ANALYSIS")
    print("=" * 60)

    print(f"\nMetadata:")
    print(f"  Tap name:     {meta.get('tap_name', meta.get('sanitized_name', 'unknown'))}")
    print(f"  Sample rate:  {meta.get('sample_rate_hz', 'unknown')} Hz")
    print(f"  Channels:     {meta.get('channels', 'unknown')}")
    print(f"  Format:       {meta.get('format', 'unknown')}")
    print(f"  Fullscale:    {meta.get('fullscale', 1.0)}")
    print(f"  Layout:       {meta.get('layout', 'unknown')}")
    print(f"  Notes:        {meta.get('notes', '')}")

    print(f"\nDuration:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Duration:      {stats['duration_sec']:.3f} sec")
    if 'frames_written' in meta:
        print(f"  Frames:        {meta['frames_written']:,}")
    if 'gap_frames' in meta and meta['gap_frames'] > 0:
        print(f"  Gap frames:    {meta['gap_frames']:,}")
        print(f"  Gaps filled:   {meta.get('gaps_filled', False)}")

    print(f"\nAmplitude statistics:")
    print(f"  Peak:      {stats['peak']:.6f}  ({stats['peak_dbfs']:.1f} dBFS)")
    print(f"  RMS:       {stats['rms']:.6f}  ({stats['rms_dbfs']:.1f} dBFS)")
    print(f"  DC offset: {stats['dc_offset']:.6f}")
    print(f"  Min:       {stats['min']:.6f}")
    print(f"  Max:       {stats['max']:.6f}")
    print(f"  Std dev:   {stats['std']:.6f}")

    print(f"\nData quality:")
    print(f"  NaN samples:     {stats['nan_count']:,}")
    print(f"  Inf samples:     {stats['inf_count']:,}")
    print(f"  Finite samples:  {stats['finite_count']:,}")
    print(f"  Clipped samples: {stats['clipped_samples']:,}")

    if 'scaling_warning' in stats:
        print(f"  WARNING: {stats['scaling_warning']}")

    if channel_stats:
        print(f"\nStereo analysis:")
        print(f"  Channel correlation: {channel_stats['channel_correlation']:.4f}")
        print(f"  (L+R) RMS:           {channel_stats['sum_rms']:.6f}")
        print(f"  (L-R) RMS:           {channel_stats['diff_rms']:.6f}")
        print(f"  Sum/Diff ratio:      {channel_stats['sum_diff_ratio']:.2f}")

        if channel_stats['sum_diff_ratio'] < 2.0:
            print("  WARNING: Low sum/diff ratio may indicate phase cancellation!")

    # Warnings
    print("\nDiagnostics:")
    warnings_found = False

    if stats['nan_count'] > 0:
        print(f"  WARNING: {stats['nan_count']} NaN values detected!")
        warnings_found = True
    if stats['inf_count'] > 0:
        print(f"  WARNING: {stats['inf_count']} Inf values detected!")
        warnings_found = True
    if stats['clipped_samples'] > 0:
        clip_pct = 100 * stats['clipped_samples'] / stats['total_samples']
        print(f"  WARNING: {clip_pct:.2f}% of samples are clipped!")
        warnings_found = True
    if abs(stats['dc_offset']) > 0.01 and stats['rms'] > 0:
        dc_ratio = abs(stats['dc_offset']) / stats['rms']
        if dc_ratio > 0.1:
            print(f"  WARNING: Significant DC offset (DC/RMS = {dc_ratio:.2f})")
            warnings_found = True
    if stats['rms_dbfs'] < -60:
        print(f"  WARNING: RMS is very low ({stats['rms_dbfs']:.1f} dBFS) - audio may be near-silent!")
        warnings_found = True
    if stats['peak_dbfs'] < -40:
        print(f"  WARNING: Peak is very low ({stats['peak_dbfs']:.1f} dBFS) - audio may be effectively silent!")
        warnings_found = True

    if not warnings_found:
        print("  No issues detected")

    print("=" * 60 + "\n")


def compare_taps(tap_files: list):
    """Compare multiple tap files side-by-side with dBFS normalization."""
    print("\n" + "=" * 70)
    print("TAP COMPARISON (all values normalized to dBFS)")
    print("=" * 70)

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
    print(f"\n{'Tap':<30} {'SR':>8} {'Ch':>4} {'Duration':>10} {'RMS dBFS':>10} {'Peak dBFS':>10} {'DC':>10} {'NaN':>6} {'Inf':>6}")
    print("-" * 114)

    for path, meta, stats in results:
        name = Path(path).stem[:28]
        print(f"{name:<30} {meta.get('sample_rate_hz', 0):>8} {meta.get('channels', 1):>4} "
              f"{stats['duration_sec']:>10.3f} {stats['rms_dbfs']:>10.1f} {stats['peak_dbfs']:>10.1f} "
              f"{stats['dc_offset']:>10.6f} {stats['nan_count']:>6} {stats['inf_count']:>6}")

    # Check for energy drop between taps (using dBFS)
    if len(results) >= 2:
        print("\nEnergy comparison (dBFS):")
        for i in range(1, len(results)):
            prev_rms_dbfs = results[i-1][2]['rms_dbfs']
            curr_rms_dbfs = results[i][2]['rms_dbfs']
            db_change = curr_rms_dbfs - prev_rms_dbfs
            print(f"  {Path(results[i-1][0]).stem} -> {Path(results[i][0]).stem}: {db_change:+.1f} dB")
            if db_change < -20:
                print(f"    WARNING: >20dB drop may indicate DSP/cancellation issue!")
            elif db_change < -10:
                print(f"    NOTE: >10dB drop - verify this is expected")


def main():
    parser = argparse.ArgumentParser(description='Analyze audio tap dumps')
    parser.add_argument('files', nargs='+', help='Raw PCM file(s) to analyze')
    parser.add_argument('--json', help='JSON sidecar file (auto-detected if not specified)')
    parser.add_argument('--output', '-o', help='Output plot filename (default: <input>_analysis.png)')
    parser.add_argument('--features', '-f', action='store_true',
                        help='Treat input as mel-feature dump')
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

        # Auto-detect features from metadata
        is_features = args.features or meta.get('kind') == 'mel_features'

        # Compute stats
        stats = compute_stats(data, meta)
        channel_stats = {}
        if not is_features and data.ndim == 2 and data.shape[1] >= 2:
            channel_stats = compute_channel_stats(data)

        # Print stats
        print_stats(stats, channel_stats, meta)

        # Generate plot
        if not args.no_plot:
            output_path = args.output or str(Path(input_file).with_suffix('')) + '_analysis.png'
            try:
                plot_audio(data, meta, output_path, is_features=is_features)
            except Exception as e:
                print(f"Error generating plot: {e}")


if __name__ == '__main__':
    main()
