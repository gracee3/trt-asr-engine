#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort


def load_raw(path: str, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if int(np.prod(shape)) != arr.size:
        raise RuntimeError(f"Size mismatch for {path}: expected {shape} ({np.prod(shape)}), got {arr.size}")
    return arr.reshape(shape)


def load_meta(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_joint(sess: ort.InferenceSession, enc: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = sess.run(None, {"encoder_output": enc, "predictor_output": pred})[0]
    # Take [B=0, T=0, U=0]
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
    ap = argparse.ArgumentParser(description="Compare joint step-0 with TRT/PT snapshots via ORT.")
    ap.add_argument("--onnx", default="tools/export_onnx/out/joint.onnx")
    ap.add_argument("--trt-dir", required=True, help="Dir containing enc_slice_trt.f32/pred_g_trt.f32/meta_trt.json")
    ap.add_argument("--pt-dir", required=True, help="Dir containing enc_slice_pt.f32/pred_g_pt.f32/meta_pt.json")
    ap.add_argument("--providers", default="cpu", help="cpu|cuda|cpu,cuda")
    args = ap.parse_args()

    providers = []
    prov = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    if "cuda" in prov:
        providers.append("CUDAExecutionProvider")
    if "cpu" in prov or not providers:
        providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(args.onnx, providers=providers)

    trt_meta = load_meta(os.path.join(args.trt_dir, "meta_trt.json"))
    pt_meta = load_meta(os.path.join(args.pt_dir, "meta_pt.json"))

    enc_trt = load_raw(os.path.join(args.trt_dir, "enc_slice_trt.f32"), tuple(trt_meta["enc_shape"]))
    pred_trt = load_raw(os.path.join(args.trt_dir, "pred_g_trt.f32"), tuple(trt_meta["pred_shape"]))
    enc_pt = load_raw(os.path.join(args.pt_dir, "enc_slice_pt.f32"), tuple(pt_meta["enc_shape"]))
    pred_pt = load_raw(os.path.join(args.pt_dir, "pred_g_pt.f32"), tuple(pt_meta["pred_shape"]))

    dur_offset = int(pt_meta.get("dur_offset", trt_meta.get("dur_offset", 0)))
    dur_bins = int(pt_meta.get("dur_bins_used", trt_meta.get("dur_bins_used", 5)))

    # 4-way swap
    out_pt_pt = run_joint(sess, enc_pt, pred_pt)
    out_trt_trt = run_joint(sess, enc_trt, pred_trt)
    out_trt_pt = run_joint(sess, enc_trt, pred_pt)
    out_pt_trt = run_joint(sess, enc_pt, pred_trt)

    print("ORT joint results (duration head):")
    describe("enc_pt + pred_pt", out_pt_pt, dur_offset, dur_bins)
    describe("enc_trt + pred_trt", out_trt_trt, dur_offset, dur_bins)
    describe("enc_trt + pred_pt", out_trt_pt, dur_offset, dur_bins)
    describe("enc_pt + pred_trt", out_pt_trt, dur_offset, dur_bins)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
