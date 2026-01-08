#!/usr/bin/env python3
import argparse
import json
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

    meta = {
        "features_shape": grab_dims("features_shape"),
        "features_valid": grab(r"\"features_valid\"\s*:\s*(\d+)", int, None),
        "features_dtype": grab(r"\"features_dtype\"\s*:\s*\"([^\"]+)\"", str, None),
        "cache_ch_shape": grab_dims("cache_ch_shape"),
        "cache_tm_shape": grab_dims("cache_tm_shape"),
        "cache_ch_dtype": grab(r"\"cache_ch_dtype\"\s*:\s*\"([^\"]+)\"", str, None),
        "cache_tm_dtype": grab(r"\"cache_tm_dtype\"\s*:\s*\"([^\"]+)\"", str, None),
        "cache_len_in": grab(r"\"cache_len_in\"\s*:\s*([-\d]+)", int, 0),
    }
    return meta


def ort_type_to_np(dtype: str):
    dtype = dtype.lower()
    if "int64" in dtype:
        return np.int64
    if "int32" in dtype:
        return np.int32
    if "float16" in dtype:
        return np.float16
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


def load_f32(path: str, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if shape and int(np.prod(shape)) != arr.size:
        raise RuntimeError(f"Size mismatch for {path}: expected {shape} ({np.prod(shape)}), got {arr.size}")
    return arr.reshape(shape) if shape else arr


def stats(diff: np.ndarray) -> Dict[str, float]:
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "p99_abs": float(np.percentile(abs_diff, 99)) if abs_diff.size else 0.0,
    }


def run_joint(sess: ort.InferenceSession, enc: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = sess.run(None, {"encoder_output": enc, "predictor_output": pred})[0]
    return out[0, 0, 0]


def top2(vec: np.ndarray) -> Tuple[int, float, int, float]:
    if vec.size < 2:
        idx = int(np.argmax(vec)) if vec.size else -1
        val = float(vec[idx]) if idx >= 0 else 0.0
        return idx, val, -1, 0.0
    idx = np.argpartition(vec, -2)[-2:]
    idx = idx[np.argsort(vec[idx])[::-1]]
    i0, i1 = int(idx[0]), int(idx[1])
    return i0, float(vec[i0]), i1, float(vec[i1])


def describe(label: str, vec: np.ndarray, dur_offset: int, dur_bins: int) -> None:
    dur = vec[dur_offset : dur_offset + dur_bins]
    i0, v0, i1, v1 = top2(dur)
    margin = v0 - v1
    print(f"{label}: best_dur_idx={i0} v0={v0:.6f} v1={v1:.6f} margin={margin:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare TRT encoder step-0 vs ORT streaming encoder.")
    ap.add_argument("--onnx", default="tools/export_onnx/out/encoder_streaming.onnx")
    ap.add_argument("--joint-onnx", default="tools/export_onnx/out/joint.onnx")
    ap.add_argument("--trt-dir", required=True, help="Dir with features_in_trt.f32/cache inputs/enc_out_t0_trt.f32")
    ap.add_argument("--pt-dir", default="", help="Optional PT snapshot dir for pred_g_pt.f32/meta_pt.json")
    ap.add_argument("--providers", default="cpu", help="cpu|cuda|cpu,cuda")
    args = ap.parse_args()

    providers = []
    prov = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    if "cuda" in prov:
        providers.append("CUDAExecutionProvider")
    if "cpu" in prov or not providers:
        providers.append("CPUExecutionProvider")

    enc_meta_path = os.path.join(args.trt_dir, "meta_enc_trt.json")
    enc_meta = parse_meta_enc(enc_meta_path)

    feat_shape = tuple(enc_meta["features_shape"])
    if not feat_shape:
        raise RuntimeError(f"Missing features_shape in {enc_meta_path}")
    T_valid = int(enc_meta.get("features_valid") or feat_shape[-1])

    feat = load_f32(os.path.join(args.trt_dir, "features_in_trt.f32"), feat_shape)
    cache_ch = load_f32(os.path.join(args.trt_dir, "cache_last_channel_in_trt.f32"),
                        tuple(enc_meta.get("cache_ch_shape") or []))
    cache_tm = load_f32(os.path.join(args.trt_dir, "cache_last_time_in_trt.f32"),
                        tuple(enc_meta.get("cache_tm_shape") or []))
    cache_len_in = int(enc_meta.get("cache_len_in") or 0)

    enc_sess = ort.InferenceSession(args.onnx, providers=providers)
    inputs = {i.name: i for i in enc_sess.get_inputs()}

    def make_len_input(name: str, value: int) -> np.ndarray:
        info = inputs[name]
        dtype = ort_type_to_np(info.type)
        shape = normalize_shape(info.shape)
        return np.full(shape, value, dtype=dtype)

    feed = {
        "audio_signal": feat,
        "length": make_len_input("length", T_valid),
        "cache_last_channel": cache_ch,
        "cache_last_time": cache_tm,
        "cache_last_channel_len": make_len_input("cache_last_channel_len", cache_len_in),
    }

    outputs = enc_sess.run(None, feed)
    out_map = {o.name: v for o, v in zip(enc_sess.get_outputs(), outputs)}

    enc_ort = out_map.get("encoder_output")
    if enc_ort is None:
        raise RuntimeError("encoder_output not found in ORT outputs")

    if enc_ort.ndim != 3:
        raise RuntimeError(f"Unexpected encoder_output shape: {enc_ort.shape}")
    enc_ort_t0 = enc_ort[0, :, 0]

    enc_trt_t0 = np.fromfile(os.path.join(args.trt_dir, "enc_out_t0_trt.f32"), dtype=np.float32)
    if enc_trt_t0.size != enc_ort_t0.size:
        raise RuntimeError(f"enc_out_t0_trt size {enc_trt_t0.size} != enc_ort_t0 size {enc_ort_t0.size}")

    enc_diff = enc_ort_t0 - enc_trt_t0
    enc_stats = stats(enc_diff)

    cache_len_out = out_map.get("cache_last_channel_len_out")
    cache_len_out_val = None
    if cache_len_out is not None:
        cache_len_out_val = int(np.array(cache_len_out).reshape(-1)[0])

    print("Encoder step-0 parity:")
    print(f"  enc_out_t0 max_abs={enc_stats['max_abs']:.6f} mean_abs={enc_stats['mean_abs']:.6f} p99_abs={enc_stats['p99_abs']:.6f}")
    print(f"  cache_in max_abs: channel={float(np.max(np.abs(cache_ch))):.6f} time={float(np.max(np.abs(cache_tm))):.6f}")
    if cache_len_out_val is None:
        print(f"  features_valid={T_valid} cache_len_in={cache_len_in}")
    else:
        print(f"  features_valid={T_valid} cache_len_in={cache_len_in} cache_len_out_ort={cache_len_out_val}")

    enc_slice_path = os.path.join(args.trt_dir, "enc_slice_trt.f32")
    meta_trt_path = os.path.join(args.trt_dir, "meta_trt.json")
    if os.path.exists(enc_slice_path) and os.path.exists(meta_trt_path):
        with open(meta_trt_path, "r", encoding="utf-8") as f:
            meta_trt = json.load(f)
        enc_shape = tuple(meta_trt["enc_shape"])
        enc_trt = load_f32(enc_slice_path, enc_shape)
        if enc_trt.shape == enc_ort.shape:
            full_stats = stats(enc_ort - enc_trt)
            print(f"  enc_full max_abs={full_stats['max_abs']:.6f} mean_abs={full_stats['mean_abs']:.6f} p99_abs={full_stats['p99_abs']:.6f}")
        else:
            print(f"  enc_full skipped (shape mismatch ort={enc_ort.shape} trt={enc_trt.shape})")

    if args.pt_dir:
        pt_meta_path = os.path.join(args.pt_dir, "meta_pt.json")
        pred_path = os.path.join(args.pt_dir, "pred_g_pt.f32")
        if os.path.exists(pt_meta_path) and os.path.exists(pred_path):
            with open(pt_meta_path, "r", encoding="utf-8") as f:
                pt_meta = json.load(f)
            pred_shape = tuple(pt_meta["pred_shape"])
            pred_pt = load_f32(pred_path, pred_shape)

            joint_sess = ort.InferenceSession(args.joint_onnx, providers=providers)
            out = run_joint(joint_sess, enc_ort, pred_pt)
            dur_offset = int(pt_meta.get("dur_offset", 0))
            dur_bins = int(pt_meta.get("dur_bins_used", 5))
            print("ORT joint (enc_ort + pred_pt):")
            describe("enc_ort + pred_pt", out, dur_offset, dur_bins)
        else:
            print("ORT joint skipped (missing pt snapshot inputs)")

    trt_len_meta = os.path.join(args.trt_dir, "cache_last_channel_len_out_trt.json")
    if os.path.exists(trt_len_meta):
        with open(trt_len_meta, "r", encoding="utf-8") as f:
            trt_meta = json.load(f)
        raw = trt_meta.get("raw")
        effective = trt_meta.get("effective")
        dtype = trt_meta.get("dtype")
        print(f"TRT cache_len_out: raw={raw} effective={effective} dtype={dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
