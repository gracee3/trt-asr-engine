#!/usr/bin/env python3
"""
Diagnostic tool for cache_last_time_out mismatch analysis.
Implements three checks to determine root cause of ORT vs PyTorch differences.
"""
import argparse
import base64
import json
import sys
from typing import Any, Dict, Iterator

import numpy as np
import onnxruntime as ort


def _decode_array(obj: Any) -> np.ndarray:
    """Decode JSONL array encoding (base64 or nested lists)."""
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, list):
        return np.asarray(obj)
    if isinstance(obj, dict):
        if "npy_path" in obj:
            return np.load(obj["npy_path"])
        if {"dtype", "shape", "data_b64"} <= set(obj.keys()):
            raw = base64.b64decode(obj["data_b64"])
            arr = np.frombuffer(raw, dtype=np.dtype(obj["dtype"]))
            return arr.reshape(obj["shape"])
    raise ValueError(f"Unsupported array encoding: {type(obj)}")


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Failed parsing JSONL at line {line_no}: {e}") from e


def make_session(onnx_path: str, providers: list[str]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def run_chunk(sess: ort.InferenceSession, audio: np.ndarray, length: np.ndarray,
              cache_ch: np.ndarray, cache_tm: np.ndarray, cache_len: np.ndarray) -> Dict[str, np.ndarray]:
    feed = {
        "audio_signal": np.ascontiguousarray(audio),
        "length": np.ascontiguousarray(length),
        "cache_last_channel": cache_ch,
        "cache_last_time": cache_tm,
        "cache_last_channel_len": cache_len,
    }
    outs = sess.run(None, feed)
    out_names = [o.name for o in sess.get_outputs()]
    return {k: v for k, v in zip(out_names, outs)}


def check1_per_axis_error(got: np.ndarray, ref: np.ndarray, axis_name: str) -> Dict:
    """
    Check 1: Is the mismatch mostly in one side of the K axis?
    Shape: [24, B, 1024, K] where K typically <=4
    Analyze errors per k=0,1,2,3 to detect padding-side issues.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 1: Per-{axis_name} axis error distribution")
    print(f"{'='*60}")

    # Assume shape is [24, B, 1024, K]
    assert got.ndim == 4 and ref.ndim == 4, "Expected 4D tensor"
    K = min(got.shape[3], ref.shape[3])

    results = {}
    for k in range(K):
        got_slice = got[:, :, :, k]
        ref_slice = ref[:, :, :, k]
        diff = np.abs(got_slice - ref_slice)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        std_abs = float(np.std(diff))

        results[f"k={k}"] = {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "std_abs": std_abs,
        }
        print(f"  k={k}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} std_abs={std_abs:.6f}")

    # Check if one k is significantly worse
    max_errors = np.array([results[f"k={k}"]["max_abs"] for k in range(K)])
    max_k = int(np.argmax(max_errors))
    min_k = int(np.argmin(max_errors))
    ratio = max_errors[max_k] / (max_errors[min_k] + 1e-12)

    print(f"\n  Analysis:")
    print(f"    Worst k: {max_k} (max_abs={max_errors[max_k]:.6f})")
    print(f"    Best k: {min_k} (max_abs={max_errors[min_k]:.6f})")
    print(f"    Ratio: {ratio:.2f}x")

    if ratio > 10:
        print(f"    ⚠️  LIKELY PADDING-SIDE ISSUE: k={max_k} has {ratio:.1f}x higher error")
        interpretation = "padding_side_mismatch"
    elif float(np.max(max_errors)) / float(np.min(max_errors) + 1e-12) < 2:
        print(f"    ✓  Errors uniformly distributed across k (not a simple padding issue)")
        interpretation = "uniform_error"
    else:
        print(f"    ?  Mixed pattern (may need deeper investigation)")
        interpretation = "mixed"

    return {"results": results, "interpretation": interpretation}


def check2_masked_error(got: np.ndarray, ref: np.ndarray, threshold: float = 1e-3) -> Dict:
    """
    Check 2: Does the mismatch live in "near-zero reference" zones?
    Compare only elements where abs(ref) > threshold.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 2: Masked error analysis (ignore near-zero regions)")
    print(f"{'='*60}")

    # Full comparison
    diff_full = np.abs(got - ref)
    max_abs_full = float(np.max(diff_full))
    mean_abs_full = float(np.mean(diff_full))

    # Masked comparison (only where ref is significant)
    mask = np.abs(ref) > threshold
    num_masked = int(np.sum(mask))
    total = ref.size
    pct_masked = 100.0 * num_masked / total

    if num_masked == 0:
        print(f"  ⚠️  WARNING: No reference values > {threshold} (all near-zero)")
        return {
            "threshold": threshold,
            "pct_significant": 0.0,
            "max_abs_full": max_abs_full,
            "mean_abs_full": mean_abs_full,
            "interpretation": "all_near_zero"
        }

    diff_masked = diff_full[mask]
    max_abs_masked = float(np.max(diff_masked))
    mean_abs_masked = float(np.mean(diff_masked))

    print(f"  Threshold: abs(ref) > {threshold}")
    print(f"  Significant elements: {num_masked:,} / {total:,} ({pct_masked:.1f}%)")
    print(f"\n  Full tensor:")
    print(f"    max_abs={max_abs_full:.6f} mean_abs={mean_abs_full:.6f}")
    print(f"\n  Masked (significant ref only):")
    print(f"    max_abs={max_abs_masked:.6f} mean_abs={mean_abs_masked:.6f}")

    reduction_ratio = max_abs_full / (max_abs_masked + 1e-12)
    print(f"\n  Analysis:")
    print(f"    Error reduction after masking: {reduction_ratio:.2f}x")

    if reduction_ratio > 10:
        print(f"    ⚠️  LIKELY PADDING JUNK: Errors collapse when ignoring near-zero ref regions")
        interpretation = "padding_junk"
    elif reduction_ratio < 1.5:
        print(f"    ✓  Errors persist in significant regions (not a padding artifact)")
        interpretation = "real_error"
    else:
        print(f"    ?  Moderate reduction (mixed signal)")
        interpretation = "mixed"

    return {
        "threshold": threshold,
        "pct_significant": pct_masked,
        "max_abs_full": max_abs_full,
        "mean_abs_full": mean_abs_full,
        "max_abs_masked": max_abs_masked,
        "mean_abs_masked": mean_abs_masked,
        "reduction_ratio": reduction_ratio,
        "interpretation": interpretation,
    }


def check3_sensitivity_test(sess: ort.InferenceSession, audio: np.ndarray, length: np.ndarray,
                            cache_ch: np.ndarray, cache_tm: np.ndarray, cache_len: np.ndarray,
                            num_perturbations: int = 3) -> Dict:
    """
    Check 3: Sensitivity test - does cache_last_time actually matter?
    Perturb cache_last_time and measure impact on encoder_output.
    """
    print(f"\n{'='*60}")
    print(f"CHECK 3: Semantic relevance (cache_last_time sensitivity)")
    print(f"{'='*60}")

    # Baseline
    baseline = run_chunk(sess, audio, length, cache_ch, cache_tm, cache_len)
    baseline_enc = baseline["encoder_output"]

    results = []

    # Test 1: Zero cache
    cache_tm_zero = np.zeros_like(cache_tm)
    out_zero = run_chunk(sess, audio, length, cache_ch, cache_tm_zero, cache_len)
    diff_zero = np.abs(out_zero["encoder_output"] - baseline_enc)
    max_abs_zero = float(np.max(diff_zero))
    mean_abs_zero = float(np.mean(diff_zero))

    print(f"\n  Test 1: Zero cache_last_time")
    print(f"    encoder_output delta: max_abs={max_abs_zero:.6f} mean_abs={mean_abs_zero:.6f}")
    results.append({"perturbation": "zeros", "max_abs": max_abs_zero, "mean_abs": mean_abs_zero})

    # Test 2-N: Random noise perturbations
    for i in range(num_perturbations):
        noise_scale = 0.1 * (i + 1)  # 0.1, 0.2, 0.3
        cache_tm_noise = cache_tm + np.random.randn(*cache_tm.shape).astype(np.float32) * noise_scale
        out_noise = run_chunk(sess, audio, length, cache_ch, cache_tm_noise, cache_len)
        diff_noise = np.abs(out_noise["encoder_output"] - baseline_enc)
        max_abs_noise = float(np.max(diff_noise))
        mean_abs_noise = float(np.mean(diff_noise))

        print(f"\n  Test {i+2}: Gaussian noise (σ={noise_scale:.1f})")
        print(f"    encoder_output delta: max_abs={max_abs_noise:.6f} mean_abs={mean_abs_noise:.6f}")
        results.append({"perturbation": f"noise_σ={noise_scale:.1f}", "max_abs": max_abs_noise, "mean_abs": mean_abs_noise})

    # Analysis
    max_perturbation_effect = max(r["max_abs"] for r in results)

    print(f"\n  Analysis:")
    print(f"    Max perturbation effect: {max_perturbation_effect:.6f}")

    if max_perturbation_effect < 1e-5:
        print(f"    ✓  CACHE NOT USED: encoder_output insensitive to cache_last_time")
        print(f"       → Mismatch is non-blocking (semantically dead tensor)")
        interpretation = "not_used"
    elif max_perturbation_effect < 1e-3:
        print(f"    ?  WEAKLY SENSITIVE: Small but measurable impact")
        interpretation = "weakly_sensitive"
    else:
        print(f"    ⚠️  STRONGLY SENSITIVE: cache_last_time materially affects output")
        print(f"       → Mismatch should be investigated and resolved")
        interpretation = "strongly_sensitive"

    return {
        "results": results,
        "max_effect": max_perturbation_effect,
        "interpretation": interpretation,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose cache_last_time_out mismatch between ORT and PyTorch reference"
    )
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--chunk", type=int, default=10, help="Chunk index to analyze")
    ap.add_argument("--providers", default="cpu", help="cpu|cuda")
    ap.add_argument("--output-json", default="", help="Write diagnostic results to JSON")
    args = ap.parse_args()

    prov = args.providers.split(",")
    providers = []
    if "cuda" in prov:
        providers.append("CUDAExecutionProvider")
    if "cpu" in prov or not providers:
        providers.append("CPUExecutionProvider")

    print(f"Loading ONNX model: {args.onnx}")
    sess = make_session(args.onnx, providers)

    print(f"Loading reference JSONL: {args.ref}")
    for i, rec in enumerate(iter_jsonl(args.ref)):
        if i == args.chunk:
            break
    else:
        print(f"ERROR: Chunk {args.chunk} not found in reference")
        return 1

    print(f"\nAnalyzing chunk {args.chunk}")
    print(f"{'='*60}\n")

    # Decode inputs
    audio = _decode_array(rec["inputs"]["audio_signal"]).astype(np.float32)
    length = _decode_array(rec["inputs"]["length"]).astype(np.int64)
    ref_cache_ch = _decode_array(rec["inputs"]["cache_last_channel"]).astype(np.float32)
    ref_cache_tm = _decode_array(rec["inputs"]["cache_last_time"]).astype(np.float32)
    ref_cache_len = _decode_array(rec["inputs"]["cache_last_channel_len"]).astype(np.int64)

    # Decode reference outputs
    ref_cache_tm_out = _decode_array(rec["outputs"]["cache_last_time_out"]).astype(np.float32)

    # Run ORT
    print("Running ORT inference...")
    got = run_chunk(sess, audio, length, ref_cache_ch, ref_cache_tm, ref_cache_len)
    got_cache_tm_out = got["cache_last_time_out"]

    # Pad to same shape for comparison
    if got_cache_tm_out.shape != ref_cache_tm_out.shape:
        print(f"Padding ORT output from {got_cache_tm_out.shape} to {ref_cache_tm_out.shape}")
        # Assume padding on axis 3 (K dimension)
        target_k = ref_cache_tm_out.shape[3]
        current_k = got_cache_tm_out.shape[3]
        if current_k < target_k:
            pad = [(0, 0)] * got_cache_tm_out.ndim
            pad[3] = (0, target_k - current_k)
            got_cache_tm_out = np.pad(got_cache_tm_out, pad, mode='constant', constant_values=0)
        elif current_k > target_k:
            got_cache_tm_out = got_cache_tm_out[:, :, :, :target_k]

    print(f"Shapes: ORT={got_cache_tm_out.shape} PyTorch={ref_cache_tm_out.shape}")

    # Run diagnostics
    diagnostics = {}

    diagnostics["check1"] = check1_per_axis_error(got_cache_tm_out, ref_cache_tm_out, "K")
    diagnostics["check2"] = check2_masked_error(got_cache_tm_out, ref_cache_tm_out, threshold=1e-3)
    diagnostics["check3"] = check3_sensitivity_test(
        sess, audio, length, ref_cache_ch, ref_cache_tm, ref_cache_len, num_perturbations=3
    )

    # Overall interpretation
    print(f"\n{'='*60}")
    print(f"OVERALL INTERPRETATION")
    print(f"{'='*60}\n")

    interp1 = diagnostics["check1"]["interpretation"]
    interp2 = diagnostics["check2"]["interpretation"]
    interp3 = diagnostics["check3"]["interpretation"]

    if interp3 == "not_used":
        print("✅ VERDICT: cache_last_time_out mismatch is NON-BLOCKING")
        print("   Reason: Encoder output is insensitive to cache_last_time variations")
        print("   Action: Treat as semantically dead tensor; no fix required for TRT integration")
    elif interp1 == "padding_side_mismatch" or interp2 == "padding_junk":
        print("⚠️  VERDICT: Likely PADDING CONVENTION mismatch")
        print("   Reason: Errors localized to specific regions or near-zero zones")
        print("   Action: Enforce explicit cache padding in runtime (zero-fill before feeding back)")
    elif interp3 == "strongly_sensitive":
        print("🔴 VERDICT: REQUIRES INVESTIGATION")
        print("   Reason: cache_last_time materially affects encoder_output")
        print("   Action: Inspect ONNX graph ops for cache_last_time computation")
    else:
        print("⚠️  VERDICT: MIXED SIGNALS (manual review needed)")
        print("   Action: Review detailed check outputs and run on multiple chunks")

    # Write JSON summary
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({
                "chunk": args.chunk,
                "providers": providers,
                "diagnostics": diagnostics,
                "summary": {
                    "check1_interpretation": interp1,
                    "check2_interpretation": interp2,
                    "check3_interpretation": interp3,
                }
            }, f, indent=2)
        print(f"\nDiagnostics written to: {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
