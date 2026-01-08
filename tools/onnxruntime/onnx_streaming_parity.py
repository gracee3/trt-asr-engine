#!/usr/bin/env python3
"""
ONNX Runtime parity testing against PyTorch reference JSONL.
Supports both functional (per-chunk stateless) and closed-loop (recurrent) modes.
"""
import argparse
import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import onnxruntime as ort


# ----------------------------
# Utilities: decoding JSONL arrays
# ----------------------------
def _decode_array(obj: Any) -> np.ndarray:
    """
    Supports common JSONL encodings:
      1) raw nested lists -> np.array
      2) dict: {"dtype": "...", "shape": [...], "data_b64": "..."}  (raw bytes)
      3) dict: {"npy_path": "..."} (external file pointer)
    """
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

    raise ValueError(f"Unsupported array encoding in JSONL: {type(obj)} keys={list(obj.keys()) if isinstance(obj, dict) else None}")


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


# ----------------------------
# Cache normalization
# ----------------------------
def pad_or_trunc_along_axis(x: np.ndarray, axis: int, target: int, pad_side: str = "right") -> np.ndarray:
    """
    Pads with zeros or truncates along 'axis' to reach 'target'.
    pad_side:
      - "right": valid data at the beginning, pad/trunc at the end
      - "left":  valid data at the end, pad/trunc at the beginning
    """
    cur = x.shape[axis]
    if cur == target:
        return x

    if cur > target:
        # truncate
        slicer = [slice(None)] * x.ndim
        if pad_side == "right":
            slicer[axis] = slice(0, target)
        else:
            slicer[axis] = slice(cur - target, cur)
        return x[tuple(slicer)]

    # pad
    pad_width = [(0, 0)] * x.ndim
    if pad_side == "right":
        pad_width[axis] = (0, target - cur)
    else:
        pad_width[axis] = (target - cur, 0)
    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=0)


@dataclass
class StreamingState:
    cache_last_channel: np.ndarray          # [B, L, C, 1024]
    cache_last_time: np.ndarray             # [B, L, 1024, K]
    cache_last_channel_len: np.ndarray      # [B]


def normalize_state_for_inputs(
    st: StreamingState,
    cache_size: int = 256,
    time_ctx: int = 4,
    pad_side: str = "right",
) -> StreamingState:
    ch = pad_or_trunc_along_axis(st.cache_last_channel, axis=2, target=cache_size, pad_side=pad_side)
    tm = pad_or_trunc_along_axis(st.cache_last_time, axis=3, target=time_ctx, pad_side=pad_side)
    return StreamingState(
        cache_last_channel=np.ascontiguousarray(ch),
        cache_last_time=np.ascontiguousarray(tm),
        cache_last_channel_len=np.ascontiguousarray(st.cache_last_channel_len),
    )


# ----------------------------
# Comparison
# ----------------------------
@dataclass
class DiffStats:
    max_abs: float
    max_rel: float
    mean_abs: float


def diff_stats(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> DiffStats:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    d = np.abs(a - b)
    max_abs = float(np.max(d)) if d.size else 0.0
    mean_abs = float(np.mean(d)) if d.size else 0.0
    denom = np.maximum(np.abs(b), eps)
    max_rel = float(np.max(d / denom)) if d.size else 0.0
    return DiffStats(max_abs=max_abs, max_rel=max_rel, mean_abs=mean_abs)


def assert_close(name: str, got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> Tuple[bool, DiffStats, str]:
    if got.shape != ref.shape:
        return False, DiffStats(np.inf, np.inf, np.inf), f"shape mismatch got={got.shape} ref={ref.shape}"

    stats = diff_stats(got, ref)
    ok = (stats.max_abs <= atol) or (stats.max_rel <= rtol)
    msg = f"{name}: max_abs={stats.max_abs:.3e} mean_abs={stats.mean_abs:.3e} max_rel={stats.max_rel:.3e}"
    return ok, stats, msg


def infer_time_valid_len(ref_cache_tm: np.ndarray, eps: float) -> int:
    if ref_cache_tm.size == 0:
        return 0
    max_per_k = np.max(np.abs(ref_cache_tm), axis=tuple(range(ref_cache_tm.ndim - 1)))
    idx = np.where(max_per_k > eps)[0]
    return int(idx[-1] + 1) if idx.size else 0


# ----------------------------
# ORT runner
# ----------------------------
def make_session(onnx_path: str, providers: list[str]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Determinism knobs (optional):
    # so.intra_op_num_threads = 1
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def run_chunk(
    sess: ort.InferenceSession,
    audio_signal: np.ndarray,              # [B, 128, T]
    length: np.ndarray,                    # [B]
    st: StreamingState,
) -> Dict[str, np.ndarray]:
    feed = {
        "audio_signal": np.ascontiguousarray(audio_signal),
        "length": np.ascontiguousarray(length),
        "cache_last_channel": st.cache_last_channel,
        "cache_last_time": st.cache_last_time,
        "cache_last_channel_len": st.cache_last_channel_len,
    }
    outs = sess.run(None, feed)
    out_names = [o.name for o in sess.get_outputs()]
    return {k: v for k, v in zip(out_names, outs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--mode", choices=["functional", "closed_loop"], default="functional")
    ap.add_argument("--providers", default="cpu", help="cpu|cuda|cpu,cuda (comma-separated)")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--cache-atol", type=float, default=1e-1, help="Cache_last_time tolerance (default: 0.1)")
    ap.add_argument("--cache-time-eps", type=float, default=1e-6, help="Threshold for inferring cache_last_time valid length")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    ap.add_argument("--dump-dir", default="", help="Dump mismatching chunk inputs/outputs as NPZ")
    ap.add_argument("--cache-size", type=int, default=256)
    ap.add_argument("--time-ctx", type=int, default=4)
    ap.add_argument("--cache-pad-side", choices=["right", "left"], default="right")
    ap.add_argument("--summary-json", default="", help="Write summary JSON to this path")
    args = ap.parse_args()

    prov = args.providers.split(",")
    prov = [p.strip().lower() for p in prov if p.strip()]
    providers = []
    if "cuda" in prov:
        providers.append("CUDAExecutionProvider")
    if "cpu" in prov or not providers:
        providers.append("CPUExecutionProvider")

    print(f"Creating ORT session with providers: {providers}")
    sess = make_session(args.onnx, providers)
    print(f"Session created successfully")

    # Print model I/O info
    print("\nModel Inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.type} {inp.shape}")
    print("\nModel Outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.type} {out.shape}")

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    # Initialize closed-loop state lazily from first record if present.
    running_state: Optional[StreamingState] = None

    total = 0
    failed = 0
    failures = []

    for i, rec in enumerate(iter_jsonl(args.ref)):
        if args.max_chunks and i >= args.max_chunks:
            break

        # Decode inputs
        audio = _decode_array(rec["inputs"]["audio_signal"]).astype(np.float32)
        length = _decode_array(rec["inputs"]["length"]).astype(np.int64)

        ref_cache_ch = _decode_array(rec["inputs"]["cache_last_channel"]).astype(np.float32)
        ref_cache_tm = _decode_array(rec["inputs"]["cache_last_time"]).astype(np.float32)
        ref_cache_len = _decode_array(rec["inputs"]["cache_last_channel_len"]).astype(np.int64)

        if args.mode == "functional":
            st_in = StreamingState(ref_cache_ch, ref_cache_tm, ref_cache_len)
        else:
            if running_state is None:
                running_state = StreamingState(ref_cache_ch, ref_cache_tm, ref_cache_len)
            st_in = running_state

        st_in = normalize_state_for_inputs(st_in, args.cache_size, args.time_ctx, args.cache_pad_side)

        got = run_chunk(sess, audio, length, st_in)

        # Reference outputs
        ref = {
            "encoder_output": _decode_array(rec["outputs"]["encoder_output"]).astype(np.float32),
            "encoded_lengths": _decode_array(rec["outputs"]["encoded_lengths"]).astype(np.int64),
            "cache_last_channel_out": _decode_array(rec["outputs"]["cache_last_channel_out"]).astype(np.float32),
            "cache_last_time_out": _decode_array(rec["outputs"]["cache_last_time_out"]).astype(np.float32),
            "cache_last_channel_len_out": _decode_array(rec["outputs"]["cache_last_channel_len_out"]).astype(np.int64),
        }

        # Normalize dynamic cache outputs from ORT to fixed max shapes before compare
        got_cache_ch = pad_or_trunc_along_axis(got["cache_last_channel_out"], axis=2, target=args.cache_size, pad_side=args.cache_pad_side)
        got_cache_tm = pad_or_trunc_along_axis(got["cache_last_time_out"], axis=3, target=args.time_ctx, pad_side=args.cache_pad_side)
        ref_cache_ch = pad_or_trunc_along_axis(ref["cache_last_channel_out"], axis=2, target=args.cache_size, pad_side=args.cache_pad_side)
        ref_cache_tm = pad_or_trunc_along_axis(ref["cache_last_time_out"], axis=3, target=args.time_ctx, pad_side=args.cache_pad_side)

        ref_cache_len = int(np.min(ref["cache_last_channel_len_out"])) if ref["cache_last_channel_len_out"].size else 0
        if ref_cache_len < 0:
            ref_cache_len = 0
        if ref_cache_len > args.cache_size:
            ref_cache_len = args.cache_size

        got_cache_ch_valid = got_cache_ch[:, :, :ref_cache_len, :] if ref_cache_len > 0 else got_cache_ch[:, :, :0, :]
        ref_cache_ch_valid = ref_cache_ch[:, :, :ref_cache_len, :] if ref_cache_len > 0 else ref_cache_ch[:, :, :0, :]

        time_valid_len = infer_time_valid_len(ref_cache_tm, args.cache_time_eps)
        got_cache_tm_valid = got_cache_tm[..., :time_valid_len] if time_valid_len > 0 else got_cache_tm[..., :0]
        ref_cache_tm_valid = ref_cache_tm[..., :time_valid_len] if time_valid_len > 0 else ref_cache_tm[..., :0]

        # Enforce output time len parity (protect valid_out_len=1)
        if got["encoder_output"].shape != ref["encoder_output"].shape:
            msg = f"encoder_output shape mismatch got={got['encoder_output'].shape} ref={ref['encoder_output'].shape}"
            print(f"[chunk {i}] FAIL: {msg}")
            failed += 1
            failures.append({"chunk": i, "error": msg})
            if args.dump_dir:
                np.savez(os.path.join(args.dump_dir, f"chunk_{i:05d}_shape_mismatch.npz"),
                         audio=audio, length=length,
                         cache_ch_in=st_in.cache_last_channel, cache_tm_in=st_in.cache_last_time, cache_len_in=st_in.cache_last_channel_len,
                         **{f"got_{k}": v for k, v in got.items()},
                         **{f"ref_{k}": v for k, v in ref.items()})
            continue

        checks = []
        checks.append(assert_close("encoder_output", got["encoder_output"], ref["encoder_output"], args.atol, args.rtol))
        checks.append((np.array_equal(got["encoded_lengths"], ref["encoded_lengths"]),
                       DiffStats(0, 0, 0),
                       f"encoded_lengths equal={np.array_equal(got['encoded_lengths'], ref['encoded_lengths'])} got={got['encoded_lengths']} ref={ref['encoded_lengths']}"))
        if ref_cache_len > 0:
            checks.append(assert_close("cache_last_channel_out", got_cache_ch_valid, ref_cache_ch_valid, args.atol, args.rtol))
        else:
            checks.append((True, DiffStats(0, 0, 0), "cache_last_channel_out skipped (valid_len=0)"))
        if time_valid_len > 0:
            checks.append(assert_close("cache_last_time_out", got_cache_tm_valid, ref_cache_tm_valid, args.cache_atol, args.rtol))
        else:
            checks.append((True, DiffStats(0, 0, 0), "cache_last_time_out skipped (valid_len=0)"))
        checks.append((np.array_equal(got["cache_last_channel_len_out"], ref["cache_last_channel_len_out"]),
                       DiffStats(0, 0, 0),
                       f"cache_last_channel_len_out equal={np.array_equal(got['cache_last_channel_len_out'], ref['cache_last_channel_len_out'])} got={got['cache_last_channel_len_out']} ref={ref['cache_last_channel_len_out']}"))

        ok_all = True
        chunk_failures = []
        for ok, stats, msg in checks:
            if not ok:
                ok_all = False
                chunk_failures.append(msg)
                print(f"[chunk {i}] FAIL: {msg}")
            else:
                print(f"[chunk {i}] PASS: {msg}")

        if not ok_all:
            failed += 1
            failures.append({"chunk": i, "errors": chunk_failures})
            if args.dump_dir:
                np.savez(os.path.join(args.dump_dir, f"chunk_{i:05d}_fail.npz"),
                         audio=audio, length=length,
                         cache_ch_in=st_in.cache_last_channel, cache_tm_in=st_in.cache_last_time, cache_len_in=st_in.cache_last_channel_len,
                         got_encoder_output=got["encoder_output"],
                         got_encoded_lengths=got["encoded_lengths"],
                         got_cache_last_channel_out=got["cache_last_channel_out"],
                         got_cache_last_time_out=got["cache_last_time_out"],
                         got_cache_last_channel_len_out=got["cache_last_channel_len_out"],
                         ref_encoder_output=ref["encoder_output"],
                         ref_encoded_lengths=ref["encoded_lengths"],
                         ref_cache_last_channel_out=ref["cache_last_channel_out"],
                         ref_cache_last_time_out=ref["cache_last_time_out"],
                         ref_cache_last_channel_len_out=ref["cache_last_channel_len_out"])

        # Update running state for closed-loop mode
        if args.mode == "closed_loop":
            running_state = StreamingState(
                cache_last_channel=np.ascontiguousarray(got_cache_ch.astype(np.float32)),
                cache_last_time=np.ascontiguousarray(got_cache_tm.astype(np.float32)),
                cache_last_channel_len=np.ascontiguousarray(got["cache_last_channel_len_out"].astype(np.int64)),
            )

        total += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Providers: {providers}")
    print(f"Total chunks: {total}")
    print(f"Passed: {total - failed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {100.0 * (total - failed) / total if total > 0 else 0:.2f}%")

    # Write summary JSON if requested
    if args.summary_json:
        summary = {
            "mode": args.mode,
            "providers": providers,
            "onnx": args.onnx,
            "ref": args.ref,
            "total_chunks": total,
            "passed": total - failed,
            "failed": failed,
            "pass_rate": 100.0 * (total - failed) / total if total > 0 else 0,
            "atol": args.atol,
            "rtol": args.rtol,
            "cache_atol": args.cache_atol,
            "failures": failures,
        }
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to: {args.summary_json}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
