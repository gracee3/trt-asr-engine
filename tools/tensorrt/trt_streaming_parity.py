#!/usr/bin/env python3
"""
TensorRT parity testing against PyTorch reference JSONL.
Supports both functional (per-chunk stateless) and closed-loop (recurrent) modes.

Based on the ORT parity harness structure with TRT-specific runtime.
Enforces streaming contract invariants (valid_out_len, cache_len bounds) at runtime.
"""
import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import tensorrt as trt
    from cuda.bindings import runtime as cudart
except ImportError:
    raise ImportError(
        "TensorRT and CUDA Python bindings required. Install with: pip install tensorrt-cu12 cuda-python"
    )


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
    """
    cur = x.shape[axis]
    if cur == target:
        return x

    if cur > target:
        slicer = [slice(None)] * x.ndim
        if pad_side == "right":
            slicer[axis] = slice(0, target)
        else:
            slicer[axis] = slice(cur - target, cur)
        return x[tuple(slicer)]

    pad_width = [(0, 0)] * x.ndim
    if pad_side == "right":
        pad_width[axis] = (0, target - cur)
    else:
        pad_width[axis] = (target - cur, 0)
    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=0)


@dataclass
class StreamingState:
    cache_last_channel: np.ndarray          # [B, 24, C, 1024]
    cache_last_time: np.ndarray             # [B, 24, 1024, K]
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
# CUDA utilities
# ----------------------------
def cuda_check(err):
    """Check CUDA error and raise if not success."""
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        err_name = str(err).split('.')[-1]
        raise RuntimeError(f"CUDA Error: {err_name}")


def cuda_malloc(size: int):
    """Allocate device memory."""
    err, ptr = cudart.cudaMalloc(size)
    cuda_check(err)
    return ptr


def cuda_free(ptr):
    """Free device memory."""
    err = cudart.cudaFree(ptr)
    cuda_check(err)


def cuda_memcpy_htod(dst, src: np.ndarray):
    """Host to device copy."""
    err = cudart.cudaMemcpy(dst, src.ctypes.data, src.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    cuda_check(err)


def cuda_memcpy_dtoh(dst: np.ndarray, src):
    """Device to host copy."""
    err = cudart.cudaMemcpy(dst.ctypes.data, src, dst.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cuda_check(err)


# ----------------------------
# TRT Runner
# ----------------------------
class TRTStreamingEncoder:
    """TensorRT streaming encoder wrapper with contract enforcement."""

    def __init__(
        self,
        engine_path: str,
        batch_size: int = 1,
        cache_size: int = 256,
        time_ctx: int = 4,
        valid_out_len_expected: Optional[int] = None,
    ):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.time_ctx = time_ctx
        self.valid_out_len_expected = valid_out_len_expected

        # Load engine
        print(f"Loading TRT engine from: {engine_path}")
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get tensor info
        self.input_names = []
        self.output_names = []
        self.tensor_info = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)

            self.tensor_info[name] = {
                "mode": mode,
                "dtype": dtype,
                "shape": shape,
            }

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"Inputs: {self.input_names}")
        print(f"Outputs: {self.output_names}")

        # Buffers will be allocated per-inference based on actual shapes
        self.d_buffers = {}
        self.h_outputs = {}

        # Timing stats
        self.inference_times = []

    def _get_numpy_dtype(self, trt_dtype):
        """Convert TRT dtype to numpy dtype."""
        dtype_map = {
            trt.float32: np.float32,
            trt.float16: np.float16,
            trt.int32: np.int32,
            trt.int64: np.int64,
            trt.int8: np.int8,
            trt.bool: np.bool_,
        }
        return dtype_map.get(trt_dtype, np.float32)

    def _allocate_buffers(self, input_shapes: Dict[str, Tuple[int, ...]]):
        """Allocate device buffers based on input shapes."""
        # Free existing buffers
        for ptr in self.d_buffers.values():
            cuda_free(ptr)
        self.d_buffers = {}
        self.h_outputs = {}

        # Set input shapes and allocate
        for name in self.input_names:
            shape = input_shapes[name]
            self.context.set_input_shape(name, shape)
            dtype = self._get_numpy_dtype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.d_buffers[name] = cuda_malloc(size)
            self.context.set_tensor_address(name, self.d_buffers[name])

        # Get output shapes and allocate
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._get_numpy_dtype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.d_buffers[name] = cuda_malloc(size)
            self.h_outputs[name] = np.zeros(shape, dtype=dtype)
            self.context.set_tensor_address(name, self.d_buffers[name])

    def infer(
        self,
        audio_signal: np.ndarray,      # [B, 128, T]
        length: np.ndarray,            # [B]
        cache_last_channel: np.ndarray,  # [B, 24, 256, 1024]
        cache_last_time: np.ndarray,     # [B, 24, 1024, 4]
        cache_last_channel_len: np.ndarray,  # [B]
    ) -> Dict[str, np.ndarray]:
        """Run inference with contract enforcement."""

        # Ensure contiguous arrays
        audio_signal = np.ascontiguousarray(audio_signal.astype(np.float32))
        length = np.ascontiguousarray(length.astype(np.int64))
        cache_last_channel = np.ascontiguousarray(cache_last_channel.astype(np.float32))
        cache_last_time = np.ascontiguousarray(cache_last_time.astype(np.float32))
        cache_last_channel_len = np.ascontiguousarray(cache_last_channel_len.astype(np.int64))

        # Allocate buffers based on actual input shapes
        input_shapes = {
            "audio_signal": audio_signal.shape,
            "length": length.shape,
            "cache_last_channel": cache_last_channel.shape,
            "cache_last_time": cache_last_time.shape,
            "cache_last_channel_len": cache_last_channel_len.shape,
        }
        self._allocate_buffers(input_shapes)

        # Copy inputs to device
        t0 = time.perf_counter()
        cuda_memcpy_htod(self.d_buffers["audio_signal"], audio_signal)
        cuda_memcpy_htod(self.d_buffers["length"], length)
        cuda_memcpy_htod(self.d_buffers["cache_last_channel"], cache_last_channel)
        cuda_memcpy_htod(self.d_buffers["cache_last_time"], cache_last_time)
        cuda_memcpy_htod(self.d_buffers["cache_last_channel_len"], cache_last_channel_len)

        # Execute
        if not self.context.execute_async_v3(0):
            raise RuntimeError("TensorRT inference failed")

        # Synchronize
        err = cudart.cudaDeviceSynchronize()
        cuda_check(err)

        # Copy outputs to host
        for name in self.output_names:
            # Reallocate output buffer with actual shape from context
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._get_numpy_dtype(self.engine.get_tensor_dtype(name))
            self.h_outputs[name] = np.zeros(shape, dtype=dtype)
            cuda_memcpy_dtoh(self.h_outputs[name], self.d_buffers[name])

        t1 = time.perf_counter()
        self.inference_times.append(t1 - t0)

        outputs = {name: self.h_outputs[name].copy() for name in self.output_names}

        # MANDATORY CONTRACT ASSERTIONS
        self._validate_streaming_contract(outputs, cache_last_channel_len)

        return outputs

    def _validate_streaming_contract(
        self,
        outputs: Dict[str, np.ndarray],
        cache_len_in: Optional[np.ndarray],
    ):
        """Hard fail on contract violations (per current contract)."""

        expected_len = self.valid_out_len_expected
        cache_len_in_val = None
        if cache_len_in is not None and cache_len_in.size:
            cache_len_in_val = int(np.min(cache_len_in))

        # Assertion 1: Encoded lengths must match expected valid_out_len (if provided)
        encoded_lengths = outputs.get("encoded_lengths")
        if encoded_lengths is not None:
            if expected_len is not None and not np.all(encoded_lengths == expected_len):
                raise RuntimeError(
                    f"STREAMING CONTRACT VIOLATION: encoded_lengths != {expected_len}, got {encoded_lengths}"
                )

        # Assertion 2: Encoder output time dimension must match expected valid_out_len (if provided)
        encoder_output = outputs.get("encoder_output")
        if encoder_output is not None:
            if expected_len is not None and encoder_output.shape[-1] != expected_len:
                raise RuntimeError(
                    f"STREAMING CONTRACT VIOLATION: encoder_output time != {expected_len}, shape={encoder_output.shape}"
                )

        # Assertion 3: Cache length must be within bounds (stateful mode)
        cache_len_out = outputs.get("cache_last_channel_len_out")
        if cache_len_out is not None:
            if np.any(cache_len_out < 0) or np.any(cache_len_out > self.cache_size):
                raise RuntimeError(
                    f"STREAMING CONTRACT VIOLATION: cache_len out of bounds (0..{self.cache_size}), got {cache_len_out}"
                )
            if cache_len_in_val is not None and np.any(cache_len_out < cache_len_in_val):
                raise RuntimeError(
                    f"STREAMING CONTRACT VIOLATION: cache_len_out < cache_len_in ({cache_len_in_val}), got {cache_len_out}"
                )

    def cleanup(self):
        """Free all allocated buffers."""
        for ptr in self.d_buffers.values():
            try:
                cuda_free(ptr)
            except:
                pass
        self.d_buffers = {}

    def __del__(self):
        self.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to TRT engine (.plan)")
    ap.add_argument("--ref", required=True, help="Path to PyTorch reference JSONL")
    ap.add_argument("--mode", choices=["functional", "closed_loop"], default="functional")
    ap.add_argument("--atol", type=float, default=5e-4, help="Absolute tolerance (default: 5e-4)")
    ap.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance (default: 1e-3)")
    ap.add_argument("--cache-atol", type=float, default=0.1, help="Cache_last_time tolerance (default: 0.1, relaxed per contract)")
    ap.add_argument("--cache-channel-atol", type=float, default=1e-2, help="Cache_last_channel tolerance (default: 1e-2)")
    ap.add_argument("--cache-time-eps", type=float, default=1e-6, help="Threshold for inferring cache_last_time valid length")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    ap.add_argument("--dump-dir", default="", help="Dump mismatching chunk inputs/outputs as NPZ")
    ap.add_argument("--cache-size", type=int, default=256)
    ap.add_argument("--time-ctx", type=int, default=4)
    ap.add_argument("--cache-pad-side", choices=["right", "left"], default="right")
    ap.add_argument("--summary-json", default="", help="Write summary JSON to this path")
    ap.add_argument("--valid-out-len", type=int, default=3, help="Expected encoded_lengths/time dim (default: 3)")
    ap.add_argument("--zero-feed-cache", action="store_true", default=False,
                    help="Always feed zero caches (diagnostic). Default: use reference caches for parity.")
    args = ap.parse_args()

    print(f"Creating TRT session from: {args.engine}")
    trt_encoder = TRTStreamingEncoder(
        args.engine,
        cache_size=args.cache_size,
        time_ctx=args.time_ctx,
        valid_out_len_expected=args.valid_out_len if args.valid_out_len > 0 else None,
    )

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    # Initialize closed-loop state lazily
    running_state: Optional[StreamingState] = None

    total = 0
    failed = 0
    failures = []
    chunk_stats = []

    for i, rec in enumerate(iter_jsonl(args.ref)):
        if args.max_chunks and i >= args.max_chunks:
            break

        # Decode inputs
        audio = _decode_array(rec["inputs"]["audio_signal"]).astype(np.float32)
        length = _decode_array(rec["inputs"]["length"]).astype(np.int64)

        ref_cache_ch = _decode_array(rec["inputs"]["cache_last_channel"]).astype(np.float32)
        ref_cache_tm = _decode_array(rec["inputs"]["cache_last_time"]).astype(np.float32)
        ref_cache_len = _decode_array(rec["inputs"]["cache_last_channel_len"]).astype(np.int64)

        B = audio.shape[0]

        if args.mode == "functional":
            st_in = StreamingState(ref_cache_ch, ref_cache_tm, ref_cache_len)
        else:
            if running_state is None:
                running_state = StreamingState(ref_cache_ch, ref_cache_tm, ref_cache_len)
            st_in = running_state

        # Optional diagnostic: zero-feed caches regardless of reference
        if args.zero_feed_cache:
            st_in = StreamingState(
                cache_last_channel=np.zeros((B, 24, args.cache_size, 1024), dtype=np.float32),
                cache_last_time=np.zeros((B, 24, 1024, args.time_ctx), dtype=np.float32),
                cache_last_channel_len=np.zeros((B,), dtype=np.int64),
            )
        else:
            st_in = normalize_state_for_inputs(st_in, args.cache_size, args.time_ctx, args.cache_pad_side)

        try:
            got = trt_encoder.infer(audio, length, st_in.cache_last_channel, st_in.cache_last_time, st_in.cache_last_channel_len)
        except RuntimeError as e:
            print(f"[chunk {i}] CONTRACT VIOLATION: {e}")
            failed += 1
            failures.append({"chunk": i, "error": str(e)})
            continue

        # Reference outputs
        ref = {
            "encoder_output": _decode_array(rec["outputs"]["encoder_output"]).astype(np.float32),
            "encoded_lengths": _decode_array(rec["outputs"]["encoded_lengths"]).astype(np.int64),
            "cache_last_channel_out": _decode_array(rec["outputs"]["cache_last_channel_out"]).astype(np.float32),
            "cache_last_time_out": _decode_array(rec["outputs"]["cache_last_time_out"]).astype(np.float32),
            "cache_last_channel_len_out": _decode_array(rec["outputs"]["cache_last_channel_len_out"]).astype(np.int64),
        }

        # Normalize dynamic cache outputs to fixed max shapes before compare
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

        # Shape check
        if got["encoder_output"].shape != ref["encoder_output"].shape:
            msg = f"encoder_output shape mismatch got={got['encoder_output'].shape} ref={ref['encoder_output'].shape}"
            print(f"[chunk {i}] FAIL: {msg}")
            failed += 1
            failures.append({"chunk": i, "error": msg})
            continue

        # Run comparisons
        checks = []

        # Primary output check (strict)
        enc_ok, enc_stats, enc_msg = assert_close("encoder_output", got["encoder_output"], ref["encoder_output"], args.atol, args.rtol)
        checks.append((enc_ok, enc_stats, enc_msg))

        # Exact match for encoded_lengths
        len_match = np.array_equal(got["encoded_lengths"], ref["encoded_lengths"])
        checks.append((len_match, DiffStats(0, 0, 0),
                       f"encoded_lengths equal={len_match} got={got['encoded_lengths']} ref={ref['encoded_lengths']}"))

        # Cache channel check (valid region only)
        if ref_cache_len > 0:
            checks.append(assert_close("cache_last_channel_out", got_cache_ch_valid, ref_cache_ch_valid, args.cache_channel_atol, args.rtol))
        else:
            checks.append((True, DiffStats(0, 0, 0), "cache_last_channel_out skipped (valid_len=0)"))

        # Cache time check (relaxed per contract; cache_last_time is numerically noisy)
        if time_valid_len > 0:
            cache_tm_ok, cache_tm_stats, cache_tm_msg = assert_close(
                "cache_last_time_out", got_cache_tm_valid, ref_cache_tm_valid, args.cache_atol, 10.0
            )
            checks.append((cache_tm_ok, cache_tm_stats, cache_tm_msg))
        else:
            checks.append((True, DiffStats(0, 0, 0), "cache_last_time_out skipped (valid_len=0)"))

        # Exact match for cache_last_channel_len_out
        cache_len_match = np.array_equal(got["cache_last_channel_len_out"], ref["cache_last_channel_len_out"])
        checks.append((cache_len_match, DiffStats(0, 0, 0),
                       f"cache_last_channel_len_out equal={cache_len_match} got={got['cache_last_channel_len_out']} ref={ref['cache_last_channel_len_out']}"))

        ok_all = True
        chunk_failures = []
        for check_ok, stats, msg in checks:
            if not check_ok:
                ok_all = False
                chunk_failures.append(msg)
                print(f"[chunk {i}] FAIL: {msg}")
            else:
                print(f"[chunk {i}] PASS: {msg}")

        # Record stats for stability analysis
        chunk_stats.append({
            "chunk": i,
            "encoder_output_max_abs": enc_stats.max_abs,
            "encoder_output_mean_abs": enc_stats.mean_abs,
            "passed": ok_all,
        })

        if not ok_all:
            failed += 1
            failures.append({"chunk": i, "errors": chunk_failures, "encoder_output_max_abs": enc_stats.max_abs})
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

    # Cleanup
    trt_encoder.cleanup()

    # Calculate timing stats
    if trt_encoder.inference_times:
        avg_time_ms = np.mean(trt_encoder.inference_times) * 1000
        p50_time_ms = np.percentile(trt_encoder.inference_times, 50) * 1000
        p95_time_ms = np.percentile(trt_encoder.inference_times, 95) * 1000
        p99_time_ms = np.percentile(trt_encoder.inference_times, 99) * 1000
    else:
        avg_time_ms = p50_time_ms = p95_time_ms = p99_time_ms = 0

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Engine: {args.engine}")
    print(f"Zero-feed cache: {args.zero_feed_cache}")
    print(f"Total chunks: {total}")
    print(f"Passed: {total - failed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {100.0 * (total - failed) / total if total > 0 else 0:.2f}%")
    print(f"\nTiming (per chunk):")
    print(f"  Mean: {avg_time_ms:.2f} ms")
    print(f"  P50:  {p50_time_ms:.2f} ms")
    print(f"  P95:  {p95_time_ms:.2f} ms")
    print(f"  P99:  {p99_time_ms:.2f} ms")

    # Stability analysis
    if chunk_stats:
        errors = [c["encoder_output_max_abs"] for c in chunk_stats]
        print(f"\nEncoder output error distribution:")
        print(f"  Min:  {min(errors):.3e}")
        print(f"  Max:  {max(errors):.3e}")
        print(f"  Mean: {np.mean(errors):.3e}")
        print(f"  P95:  {np.percentile(errors, 95):.3e}")
        print(f"  P99:  {np.percentile(errors, 99):.3e}")

    # Write summary JSON if requested
    if args.summary_json:
        summary = {
            "mode": args.mode,
            "engine": args.engine,
            "ref": args.ref,
            "zero_feed_cache": args.zero_feed_cache,
            "total_chunks": total,
            "passed": total - failed,
            "failed": failed,
            "pass_rate": 100.0 * (total - failed) / total if total > 0 else 0,
            "atol": args.atol,
            "rtol": args.rtol,
            "cache_atol": args.cache_atol,
            "cache_channel_atol": args.cache_channel_atol,
            "timing_ms": {
                "mean": avg_time_ms,
                "p50": p50_time_ms,
                "p95": p95_time_ms,
                "p99": p99_time_ms,
            },
            "encoder_output_error_distribution": {
                "min": min(errors) if chunk_stats else None,
                "max": max(errors) if chunk_stats else None,
                "mean": float(np.mean(errors)) if chunk_stats else None,
                "p95": float(np.percentile(errors, 95)) if chunk_stats else None,
                "p99": float(np.percentile(errors, 99)) if chunk_stats else None,
            } if chunk_stats else None,
            "chunk_stats": chunk_stats,
            "failures": failures,
        }
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to: {args.summary_json}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
