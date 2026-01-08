#!/usr/bin/env python3
import argparse
import json
import os
import wave
from typing import Dict, List, Tuple

import numpy as np
import torch
import nemo.collections.asr as nemo_asr


def load_model(model_arg: str):
    if os.path.exists(model_arg):
        try:
            return nemo_asr.models.ASRModel.restore_from(model_arg)
        except Exception:
            return nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_arg)
    return nemo_asr.models.ASRModel.from_pretrained(model_arg)


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        if wf.getnchannels() != 1:
            raise RuntimeError(f"WAV must be mono: channels={wf.getnchannels()}")
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise RuntimeError(f"WAV must be 16-bit PCM: sampwidth={sampwidth}")
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def load_features_raw(path: str, n_mels: int, layout: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % n_mels != 0:
        raise RuntimeError(f"features size {raw.size} not divisible by n_mels={n_mels}")
    t = raw.size // n_mels
    if layout == "frames_major":
        return raw.reshape(t, n_mels).T
    return raw.reshape(n_mels, t)


def load_chunk_frames_list(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [int(v) for v in payload]
    if isinstance(payload, dict) and "chunk_frames" in payload:
        return [int(v) for v in payload["chunk_frames"]]
    raise RuntimeError(f"Unsupported chunk_frames payload in {path}")


def try_override_normalize(model, mode: str) -> Tuple[bool, str]:
    msg = ""
    changed = False
    for obj_name in ["preprocessor", "cfg.preprocessor"]:
        obj = model
        try:
            for part in obj_name.split("."):
                obj = getattr(obj, part)
        except Exception:
            continue
        if obj is None:
            continue
        if hasattr(obj, "normalize"):
            prev = getattr(obj, "normalize")
            try:
                setattr(obj, "normalize", mode)
                msg = f"normalize {prev} -> {mode} on {obj_name}"
                changed = True
                break
            except Exception:
                pass
    return changed, msg


def extract_features(model, audio: np.ndarray, sr: int, device: torch.device) -> np.ndarray:
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
    length = torch.tensor([audio.shape[0]], dtype=torch.long, device=device)
    if sr != 16000:
        raise RuntimeError(f"Expected 16kHz audio, got {sr}")
    with torch.inference_mode():
        feats, feat_len = model.preprocessor(input_signal=audio_t, length=length)
    feats = feats.detach().cpu().numpy()
    if feats.ndim != 3:
        raise RuntimeError(f"Unexpected feature shape: {feats.shape}")
    # Normalize to [C, T]
    if feats.shape[1] < feats.shape[2]:
        feat_ct = feats[0]
    else:
        feat_ct = feats[0].T
    return feat_ct.astype(np.float32, copy=False)


def diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = np.abs(a - b)
    return {
        "max_abs": float(np.max(d)) if d.size else 0.0,
        "mean_abs": float(np.mean(d)) if d.size else 0.0,
        "p99_abs": float(np.percentile(d, 99)) if d.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare runtime dumped features vs NeMo preprocessor features.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--runtime-features", required=True, help="f32le features [C,T] bins_major by default")
    ap.add_argument("--runtime-layout", choices=["bins_major", "frames_major"], default="bins_major")
    ap.add_argument("--chunk-frames-list", default="", help="JSON list of per-chunk frame counts")
    ap.add_argument("--normalize", choices=["none", "per_feature"], default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    model = load_model(args.model)
    model.eval()
    model = model.to(device)
    if hasattr(model, "preprocessor"):
        try:
            model.preprocessor = model.preprocessor.to(device)
        except Exception:
            pass

    n_mels = int(model.cfg.preprocessor.get("features", 128))
    if args.normalize:
        ok, msg = try_override_normalize(model, args.normalize)
        print(f"[compare_features] normalize_override ok={ok} msg={msg}")

    audio, sr = load_wav(args.wav)
    nemo_feat = extract_features(model, audio, sr, device)
    runtime_feat = load_features_raw(args.runtime_features, n_mels, args.runtime_layout)

    total_t = min(nemo_feat.shape[1], runtime_feat.shape[1])
    nemo_feat = nemo_feat[:, :total_t]
    runtime_feat = runtime_feat[:, :total_t]

    print(f"[compare_features] n_mels={n_mels} nemo_T={nemo_feat.shape[1]} runtime_T={runtime_feat.shape[1]}")
    overall = diff_stats(runtime_feat, nemo_feat)
    print(f"[compare_features] overall max_abs={overall['max_abs']:.6g} mean_abs={overall['mean_abs']:.6g} p99_abs={overall['p99_abs']:.6g}")

    if not args.chunk_frames_list:
        return 0

    frames = load_chunk_frames_list(args.chunk_frames_list)
    offset = 0
    for idx, span in enumerate(frames):
        if offset >= total_t:
            break
        end = min(offset + span, total_t)
        rt = runtime_feat[:, offset:end]
        nm = nemo_feat[:, offset:end]
        stats = diff_stats(rt, nm)
        print(f"[compare_features] chunk={idx} frames={end - offset} max_abs={stats['max_abs']:.6g} mean_abs={stats['mean_abs']:.6g} p99_abs={stats['p99_abs']:.6g}")
        offset = end

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
