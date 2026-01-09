#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from omegaconf import OmegaConf


def load_model(model_arg: str):
    if os.path.exists(model_arg):
        try:
            return nemo_asr.models.ASRModel.restore_from(model_arg)
        except Exception:
            return nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_arg)
    return nemo_asr.models.ASRModel.from_pretrained(model_arg)


def to_container(cfg: Any) -> Dict[str, Any]:
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return dict(cfg) if isinstance(cfg, dict) else {"value": str(cfg)}


def safe_value(val: Any) -> Any:
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [safe_value(v) for v in val]
    if isinstance(val, dict):
        return {k: safe_value(v) for k, v in val.items()}
    if isinstance(val, torch.Tensor):
        if val.numel() <= 16:
            return val.detach().cpu().flatten().tolist()
        return f"Tensor(shape={tuple(val.shape)}, dtype={val.dtype})"
    if isinstance(val, np.ndarray):
        if val.size <= 16:
            return val.flatten().tolist()
        return f"ndarray(shape={val.shape}, dtype={val.dtype})"
    return str(val)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump NeMo preprocessor config from a .nemo model.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model = load_model(args.model)
    cfg = getattr(model, "cfg", None)
    if cfg is None or not hasattr(cfg, "preprocessor"):
        raise RuntimeError("Model config missing preprocessor section")

    pre = to_container(cfg.preprocessor)
    pre_mod = getattr(model, "preprocessor", None)
    mod_attrs = {}
    featurizer_attrs = {}
    if pre_mod is not None:
        for name in [
            "normalize",
            "log",
            "log_zero_guard_type",
            "log_zero_guard_value",
            "power",
            "preemph",
            "dither",
            "pad_to",
            "pad_value",
            "window",
            "window_size",
            "window_stride",
            "n_fft",
            "features",
            "center",
            "pad_mode",
        ]:
            if hasattr(pre_mod, name):
                try:
                    mod_attrs[name] = safe_value(getattr(pre_mod, name))
                except Exception:
                    pass
        feat = getattr(pre_mod, "featurizer", None)
        if feat is not None:
            for name in [
                "mel_scale",
                "mel_norm",
                "f_min",
                "f_max",
                "mag_power",
                "use_log",
                "log_zero_guard_type",
                "log_zero_guard_value",
                "power",
                "preemph",
                "dither",
                "n_fft",
                "n_mels",
                "hop_length",
                "win_length",
                "pad_to",
                "pad_value",
                "window",
                "center",
                "pad_mode",
            ]:
                if hasattr(feat, name):
                    try:
                        featurizer_attrs[name] = safe_value(getattr(feat, name))
                    except Exception:
                        pass

    # Derive sample-based sizes if possible.
    sr = pre.get("sample_rate", 16000)
    win_sec = pre.get("window_size", pre.get("win_length", None))
    hop_sec = pre.get("window_stride", pre.get("hop_length", None))

    derived = {
        "sample_rate": sr,
        "window_size_s": win_sec,
        "window_stride_s": hop_sec,
        "win_length_samples": int(round(win_sec * sr)) if isinstance(win_sec, (int, float)) else None,
        "hop_length_samples": int(round(hop_sec * sr)) if isinstance(hop_sec, (int, float)) else None,
    }

    payload = {
        "preprocessor": pre,
        "module_attrs": mod_attrs,
        "featurizer_attrs": featurizer_attrs,
        "derived": derived,
    }

    out_path = args.out
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote={out_path}")
    else:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
