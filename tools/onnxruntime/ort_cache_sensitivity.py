#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


def parse_dims(text: str) -> List[int]:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    parts = re.split(r"[x,]", text)
    dims = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        dims.append(int(p))
    return dims


def parse_meta_enc(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    def grab(pattern: str, cast=int, default=None):
        m = re.search(pattern, raw)
        if not m:
            return default
        return cast(m.group(1))

    def grab_dims(key: str) -> List[int]:
        m = re.search(rf"\"{re.escape(key)}\"\s*:\s*(\[[^\]]+\])", raw)
        if not m:
            return []
        return parse_dims(m.group(1))

    return {
        "features_shape": grab_dims("features_shape"),
        "features_valid": grab(r"\"features_valid\"\s*:\s*(\d+)", int, None),
        "cache_ch_shape": grab_dims("cache_ch_shape"),
        "cache_tm_shape": grab_dims("cache_tm_shape"),
    }


def load_f32(path: str, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if shape and int(np.prod(shape)) != arr.size:
        raise RuntimeError(f"Size mismatch for {path}: expected {shape} ({np.prod(shape)}), got {arr.size}")
    return arr.reshape(shape) if shape else arr


def ort_type_to_np(dtype: str):
    dtype = dtype.lower()
    if "int64" in dtype:
        return np.int64
    if "int32" in dtype:
        return np.int32
    return np.float32


def normalize_shape(shape) -> Tuple[int, ...]:
    if not shape:
        return (1,)
    out = []
    for d in shape:
        if d is None or isinstance(d, str):
            out.append(1)
        else:
            out.append(int(d))
    return tuple(out)


def run_encoder(sess: ort.InferenceSession,
                features: np.ndarray,
                length: int,
                cache_ch: np.ndarray,
                cache_tm: np.ndarray,
                cache_len: int) -> Dict[str, np.ndarray]:
    inputs = {i.name: i for i in sess.get_inputs()}

    def make_len_input(name: str, value: int) -> np.ndarray:
        info = inputs[name]
        dtype = ort_type_to_np(info.type)
        shape = normalize_shape(info.shape)
        return np.full(shape, value, dtype=dtype)

    feed = {
        "audio_signal": features,
        "length": make_len_input("length", length),
        "cache_last_channel": cache_ch,
        "cache_last_time": cache_tm,
        "cache_last_channel_len": make_len_input("cache_last_channel_len", cache_len),
    }
    outputs = sess.run(None, feed)
    return {o.name: v for o, v in zip(sess.get_outputs(), outputs)}


def stats(diff: np.ndarray) -> Dict[str, float]:
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
    }


def max_abs(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def main() -> int:
    ap = argparse.ArgumentParser(description="ORT cache sensitivity test for encoder_streaming.onnx")
    ap.add_argument("--onnx", default="tools/export_onnx/out/encoder_streaming.onnx")
    ap.add_argument("--snapshot-dir", required=True, help="Dir containing features_in_trt.f32 and meta_enc_trt.json")
    ap.add_argument("--providers", default="cpu", help="cpu|cuda|cpu,cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-len", type=int, default=64, help="cache_len value for non-zero cache test")
    args = ap.parse_args()

    providers = []
    prov = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    if "cuda" in prov:
        providers.append("CUDAExecutionProvider")
    if "cpu" in prov or not providers:
        providers.append("CPUExecutionProvider")

    meta_path = os.path.join(args.snapshot_dir, "meta_enc_trt.json")
    meta = parse_meta_enc(meta_path)
    feat_shape = tuple(meta.get("features_shape") or [])
    if not feat_shape:
        raise RuntimeError(f"Missing features_shape in {meta_path}")

    features = load_f32(os.path.join(args.snapshot_dir, "features_in_trt.f32"), feat_shape)
    length = int(meta.get("features_valid") or feat_shape[-1])

    cache_ch_shape = tuple(meta.get("cache_ch_shape") or [])
    cache_tm_shape = tuple(meta.get("cache_tm_shape") or [])
    if not cache_ch_shape or not cache_tm_shape:
        raise RuntimeError("Missing cache shapes in meta_enc_trt.json")

    cache_size = cache_ch_shape[2] if len(cache_ch_shape) > 2 else 0
    cache_len_b = min(args.cache_len, cache_size) if cache_size else args.cache_len

    rng = np.random.default_rng(args.seed)
    cache_ch_zero = np.zeros(cache_ch_shape, dtype=np.float32)
    cache_tm_zero = np.zeros(cache_tm_shape, dtype=np.float32)
    cache_ch_rand = rng.normal(0.0, 0.5, size=cache_ch_shape).astype(np.float32)
    cache_tm_rand = rng.normal(0.0, 0.5, size=cache_tm_shape).astype(np.float32)

    sess = ort.InferenceSession(args.onnx, providers=providers)
    out_a = run_encoder(sess, features, length, cache_ch_zero, cache_tm_zero, 0)
    out_b = run_encoder(sess, features, length, cache_ch_rand, cache_tm_rand, cache_len_b)

    def cache_len_val(out: Dict[str, np.ndarray]) -> int:
        v = out.get("cache_last_channel_len_out")
        if v is None:
            return 0
        return int(np.array(v).reshape(-1)[0])
    def enc_len_val(out: Dict[str, np.ndarray]) -> int:
        v = out.get("encoded_lengths")
        if v is None:
            return 0
        return int(np.array(v).reshape(-1)[0])

    enc_diff = out_b["encoder_output"] - out_a["encoder_output"]
    ch_diff = out_b["cache_last_channel_out"] - out_a["cache_last_channel_out"]
    tm_diff = out_b["cache_last_time_out"] - out_a["cache_last_time_out"]

    print("ORT cache sensitivity:")
    print(f"  cache_len_in A=0 B={cache_len_b}")
    print(f"  encoded_lengths A={enc_len_val(out_a)} B={enc_len_val(out_b)}")
    print(f"  cache_len_out A={cache_len_val(out_a)} B={cache_len_val(out_b)}")
    print(f"  encoder_output diff max_abs={stats(enc_diff)['max_abs']:.6f} mean_abs={stats(enc_diff)['mean_abs']:.6f}")
    print(f"  cache_ch_out A max_abs={max_abs(out_a['cache_last_channel_out']):.6f}")
    print(f"  cache_tm_out A max_abs={max_abs(out_a['cache_last_time_out']):.6f}")
    print(f"  cache_ch_out B max_abs={max_abs(out_b['cache_last_channel_out']):.6f}")
    print(f"  cache_tm_out B max_abs={max_abs(out_b['cache_last_time_out']):.6f}")
    print(f"  cache_ch_out diff max_abs={stats(ch_diff)['max_abs']:.6f} mean_abs={stats(ch_diff)['mean_abs']:.6f}")
    print(f"  cache_tm_out diff max_abs={stats(tm_diff)['max_abs']:.6f} mean_abs={stats(tm_diff)['mean_abs']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
