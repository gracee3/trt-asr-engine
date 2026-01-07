#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import nemo.collections.asr as nemo_asr
from contextlib import contextmanager


def _load_model(model_arg: str):
    if os.path.exists(model_arg):
        try:
            return nemo_asr.models.ASRModel.restore_from(model_arg)
        except Exception:
            return nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_arg)
    return nemo_asr.models.ASRModel.from_pretrained(model_arg)


def _onnx_input_dims(path: str, name: str) -> list[Optional[int]]:
    model = onnx.load(path, load_external_data=False)
    for inp in model.graph.input:
        if inp.name == name:
            dims = []
            t = inp.type.tensor_type
            for d in t.shape.dim:
                if d.dim_value:
                    dims.append(int(d.dim_value))
                else:
                    dims.append(None)
            return dims
    raise RuntimeError(f"Input {name} not found in {path}")


def _infer_decoder_l_h(model) -> Tuple[int, int]:
    num_layers, hidden_size = 2, 640
    try:
        decoder = model.decoder
        pred = getattr(decoder, "prediction", None)
        if pred is not None and hasattr(pred, "__getitem__"):
            dec_rnn = pred["dec_rnn"]
            lstm = getattr(dec_rnn, "lstm", None)
            if lstm is not None:
                num_layers = int(getattr(lstm, "num_layers", num_layers) or num_layers)
                hidden_size = int(getattr(lstm, "hidden_size", hidden_size) or hidden_size)
    except Exception:
        pass
    return num_layers, hidden_size


def _resolve_dim(value: Optional[int], fallback: int) -> int:
    return int(value) if value is not None and value > 0 else int(fallback)


def _diff_stats(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    d = np.abs(a - b)
    max_abs = float(np.max(d)) if d.size else 0.0
    mean_abs = float(np.mean(d)) if d.size else 0.0
    denom = np.maximum(np.abs(b), eps)
    max_rel = float(np.max(d / denom)) if d.size else 0.0
    return {"max_abs": max_abs, "mean_abs": mean_abs, "max_rel": max_rel}


def _check_close(name: str, got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> Tuple[bool, Dict[str, float]]:
    if got.shape != ref.shape:
        return False, {"max_abs": float("inf"), "mean_abs": float("inf"), "max_rel": float("inf")}
    stats = _diff_stats(got, ref)
    ok = (stats["max_abs"] <= atol) or (stats["max_rel"] <= rtol)
    return ok, stats


def _load_contract_heads(path: str) -> Optional[Dict[str, int]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        out = data["joint"]["io"]["outputs"][0]
        token = out["token_head"]
        dur = out["duration_head"]
        return {
            "token_offset": int(token["offset"]),
            "token_size": int(token["size"]),
            "dur_offset": int(dur["offset"]),
            "dur_size": int(dur["size"]),
        }
    except Exception:
        pass
    try:
        arch = data["joint"]["architecture"]
        token_size = int(arch["token_head_size"])
        dur_size = len(arch["duration_values"])
        return {
            "token_offset": 0,
            "token_size": token_size,
            "dur_offset": token_size,
            "dur_size": dur_size,
        }
    except Exception:
        return None


def _make_session(path: str, providers: list[str]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=providers)


@contextmanager
def _disable_joint_fuse_loss_wer(joint: torch.nn.Module):
    orig = None
    try:
        orig = getattr(joint, "_fuse_loss_wer", None)
        if hasattr(joint, "set_fuse_loss_wer"):
            joint.set_fuse_loss_wer(False)
        elif orig is not None:
            setattr(joint, "_fuse_loss_wer", False)
        yield
    finally:
        if orig is not None:
            try:
                if hasattr(joint, "set_fuse_loss_wer"):
                    joint.set_fuse_loss_wer(bool(orig))
                else:
                    setattr(joint, "_fuse_loss_wer", orig)
            except Exception:
                pass


@contextmanager
def _force_joint_logits(joint: torch.nn.Module):
    orig = None
    try:
        if hasattr(joint, "log_softmax"):
            orig = getattr(joint, "log_softmax", None)
            setattr(joint, "log_softmax", False)
        yield
    finally:
        if orig is not None and hasattr(joint, "log_softmax"):
            try:
                setattr(joint, "log_softmax", orig)
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Predictor + Joint parity vs PyTorch (logits-level).")
    ap.add_argument("--model", default="models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo")
    ap.add_argument("--onnx-dir", default="tools/export_onnx/out")
    ap.add_argument("--predictor-onnx", default="")
    ap.add_argument("--joint-onnx", default="")
    ap.add_argument("--contract", default="contracts/parakeet-tdt-0.6b-v3.contract.json")
    ap.add_argument("--device", default="cpu", help="torch device: cpu or cuda")
    ap.add_argument("--providers", default="cpu", help="onnxruntime providers: cpu|cuda|cpu,cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--token-id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    predictor_path = args.predictor_onnx or os.path.join(args.onnx_dir, "predictor.onnx")
    joint_path = args.joint_onnx or os.path.join(args.onnx_dir, "joint.onnx")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = _load_model(args.model)
    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    providers = []
    prov = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    if "cuda" in prov:
        providers.append("CUDAExecutionProvider")
    if "cpu" in prov or not providers:
        providers.append("CPUExecutionProvider")

    pred_sess = _make_session(predictor_path, providers)
    joint_sess = _make_session(joint_path, providers)

    h_dims = _onnx_input_dims(predictor_path, "h")
    c_dims = _onnx_input_dims(predictor_path, "c")
    l_fallback, h_fallback = _infer_decoder_l_h(model)
    num_layers = _resolve_dim(h_dims[0] if len(h_dims) > 0 else None, l_fallback)
    hidden_size = _resolve_dim(h_dims[2] if len(h_dims) > 2 else None, h_fallback)
    batch = args.batch

    y = torch.full((batch, 1), int(args.token_id), dtype=torch.long, device=device)
    h = torch.zeros(num_layers, batch, hidden_size, dtype=torch.float32, device=device)
    c = torch.zeros(num_layers, batch, hidden_size, dtype=torch.float32, device=device)

    results: Dict[str, Any] = {"predictor": {}, "joint": {}}
    failures = []

    with torch.inference_mode():
        g_pt, state = model.decoder.predict(y=y, state=[h, c], add_sos=False)
        g_pt = g_pt.transpose(1, 2).contiguous()
        h_pt, c_pt = state[0].contiguous(), state[1].contiguous()

    y_np = y.detach().cpu().numpy().astype(np.int64)
    h_np = h.detach().cpu().numpy()
    c_np = c.detach().cpu().numpy()

    g_ort, h_ort, c_ort = pred_sess.run(None, {"y": y_np, "h": h_np, "c": c_np})

    pred_pairs = {
        "g": (g_ort, g_pt.detach().cpu().numpy()),
        "h_out": (h_ort, h_pt.detach().cpu().numpy()),
        "c_out": (c_ort, c_pt.detach().cpu().numpy()),
    }
    for name, (ort_val, pt_val) in pred_pairs.items():
        ok, stats = _check_close(name, ort_val, pt_val, args.atol, args.rtol)
        results["predictor"][name] = {"ok": ok, "stats": stats}
        if not ok:
            failures.append(f"predictor {name}")

    joint_heads = _load_contract_heads(args.contract)
    enc_dims = _onnx_input_dims(joint_path, "encoder_output")
    pred_dims = _onnx_input_dims(joint_path, "predictor_output")
    d_enc = _resolve_dim(enc_dims[1] if len(enc_dims) > 1 else None, 1024)
    d_pred = _resolve_dim(pred_dims[1] if len(pred_dims) > 1 else None, 640)

    enc_np = np.random.randn(batch, d_enc, 1).astype(np.float32)
    pred_np = g_pt.detach().cpu().numpy().astype(np.float32)
    if pred_np.shape[1] != d_pred:
        raise RuntimeError(f"predictor_output dim mismatch: expected {d_pred}, got {pred_np.shape[1]}")

    with torch.inference_mode():
        enc_t = torch.from_numpy(enc_np).to(device)
        pred_t = torch.from_numpy(pred_np).to(device)
        with _disable_joint_fuse_loss_wer(model.joint), _force_joint_logits(model.joint):
            joint_pt = model.joint(encoder_outputs=enc_t, decoder_outputs=pred_t)
        joint_pt = joint_pt.detach().cpu().numpy()

    joint_ort = joint_sess.run(None, {"encoder_output": enc_np, "predictor_output": pred_np})[0]

    ok, stats = _check_close("joint_output", joint_ort, joint_pt, args.atol, args.rtol)
    results["joint"]["joint_output"] = {"ok": ok, "stats": stats}
    if not ok:
        failures.append("joint_output")

    if joint_heads:
        t0 = joint_heads["token_offset"]
        t1 = t0 + joint_heads["token_size"]
        d0 = joint_heads["dur_offset"]
        d1 = d0 + joint_heads["dur_size"]

        pt_vec = joint_pt[0, 0, 0]
        ort_vec = joint_ort[0, 0, 0]
        pt_tok = int(np.argmax(pt_vec[t0:t1]))
        ort_tok = int(np.argmax(ort_vec[t0:t1]))
        pt_dur = int(np.argmax(pt_vec[d0:d1]))
        ort_dur = int(np.argmax(ort_vec[d0:d1]))
        results["joint"]["argmax"] = {
            "token": {"pt": pt_tok, "ort": ort_tok, "match": pt_tok == ort_tok},
            "duration": {"pt": pt_dur, "ort": ort_dur, "match": pt_dur == ort_dur},
        }
        if pt_tok != ort_tok:
            failures.append("joint token argmax")
        if pt_dur != ort_dur:
            failures.append("joint duration argmax")

    print(json.dumps(results, indent=2))

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if failures:
        print(f"FAIL: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
