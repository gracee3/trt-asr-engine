#!/usr/bin/env python3
"""Plot encoder_output error over chunks to verify no accumulation."""
import argparse
import json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--output-png", default="trt_stability_plot.png")
    args = ap.parse_args()

    with open(args.summary_json) as f:
        data = json.load(f)

    chunk_stats = data.get("chunk_stats", [])
    if not chunk_stats:
        print("No chunk_stats found in summary JSON")
        return

    chunks = [c["chunk"] for c in chunk_stats]
    errors = [c["encoder_output_max_abs"] for c in chunk_stats]

    # Calculate trend line
    z = np.polyfit(chunks, errors, 1)
    slope = z[0]
    trend_line = np.poly1d(z)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(chunks, errors, 'b.', alpha=0.5, markersize=3, label='Per-chunk error')
        ax.plot(chunks, trend_line(chunks), 'r-', linewidth=2, label=f'Trend (slope={slope:.2e})')
        ax.axhline(y=5e-4, color='orange', linestyle='--', linewidth=1.5, label='P95 target (5e-4)')
        ax.axhline(y=1e-3, color='red', linestyle='--', linewidth=1.5, label='Max target (1e-3)')
        ax.set_xlabel('Chunk Index')
        ax.set_ylabel('encoder_output max_abs error')
        ax.set_title(f'TRT Closed-Loop Stability ({len(chunks)} chunks)\nSlope: {slope:.2e} (should be ~0)')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output_png, dpi=150)
        print(f"Plot saved to: {args.output_png}")
    except ImportError:
        print("matplotlib not available, printing text summary instead")

    # Text summary
    print(f"\n=== Stability Analysis ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Error min:  {min(errors):.3e}")
    print(f"Error max:  {max(errors):.3e}")
    print(f"Error mean: {np.mean(errors):.3e}")
    print(f"Error P95:  {np.percentile(errors, 95):.3e}")
    print(f"Error P99:  {np.percentile(errors, 99):.3e}")
    print(f"Trend slope: {slope:.3e}")
    if abs(slope) < 1e-6:
        print("Slope ~0: NO ERROR ACCUMULATION")
    elif slope > 0:
        print(f"WARNING: Positive slope indicates potential error growth")
    else:
        print(f"Negative slope (errors decreasing)")


if __name__ == "__main__":
    main()
