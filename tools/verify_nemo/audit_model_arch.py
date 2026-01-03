#!/usr/bin/env python3
import argparse
import dataclasses
import json
import os
import sys
import inspect

import torch
import nemo.collections.asr as nemo_asr

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional dependency
    OmegaConf = None


def _cfg_get(cfg, path, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur if cur is not None else default


def _to_container(cfg):
    if cfg is None:
        return None
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    if OmegaConf is None:
        return str(cfg)
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return str(cfg)


def _pick_first_value(cfg, keys):
    for key in keys:
        val = _cfg_get(cfg, key, None)
        if val is not None:
            return val
    return None


def _collect_conv_stats(module):
    convs = []
    depthwise = []
    subsample = []
    for name, m in module.named_modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
            kernel = list(m.kernel_size) if hasattr(m, "kernel_size") else None
            stride = list(m.stride) if hasattr(m, "stride") else None
            groups = int(getattr(m, "groups", 1) or 1)
            in_ch = int(getattr(m, "in_channels", 0) or 0)
            out_ch = int(getattr(m, "out_channels", 0) or 0)
            entry = {
                "name": name,
                "type": m.__class__.__name__,
                "kernel_size": kernel,
                "stride": stride,
                "groups": groups,
                "in_channels": in_ch,
                "out_channels": out_ch,
            }
            convs.append(entry)
            if groups == in_ch and in_ch == out_ch and in_ch > 0:
                depthwise.append(entry)
            if "subsampling" in name.lower() or "subsample" in name.lower():
                subsample.append(entry)
    return {
        "all": convs,
        "depthwise": depthwise,
        "subsampling": subsample,
    }


def _call_encoder_forward(encoder, x, x_len):
    fn = encoder.forward
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    names = [p.name for p in params]

    def _pick(candidates):
        for name in candidates:
            if name in names:
                return name
        return None

    audio_name = _pick(["audio_signal", "x", "input_signal", "features"])
    length_name = _pick(["length", "x_len", "input_length", "audio_signal_length", "feature_length"])
    kwargs = {}
    if audio_name:
        kwargs[audio_name] = x
    if length_name:
        kwargs[length_name] = x_len

    missing = [
        p.name
        for p in params
        if p.default is inspect._empty and p.name not in kwargs
    ]
    if missing:
        raise RuntimeError(f"encoder.forward missing required args {missing}; signature={sig}")
    return fn(**kwargs)


def _infer_subsampling_factor(encoder, n_mels, device, time_steps):
    x = torch.randn(1, n_mels, time_steps, device=device)
    x_len = torch.tensor([time_steps], dtype=torch.int64, device=device)
    with torch.no_grad():
        out = _call_encoder_forward(encoder, x, x_len)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError(f"Unexpected encoder output structure: {type(out)}")
    out_len = out[1]
    if not torch.is_tensor(out_len):
        raise RuntimeError("encoder output length is not a tensor")
    denom = int(out_len.item())
    if denom <= 0:
        raise RuntimeError("encoder output length <= 0")
    return float(time_steps) / float(denom)


def _status_from_expected(value, expected):
    if expected is None:
        return "warn"
    if value is None:
        return "warn"
    if isinstance(expected, (list, tuple, set)):
        return "pass" if value in expected else "warn"
    return "pass" if value == expected else "warn"


def load_model(model_arg):
    if os.path.exists(model_arg):
        try:
            return nemo_asr.models.ASRModel.restore_from(model_arg)
        except Exception as exc:
            print(f"ASRModel.restore_from failed: {exc}")
            return nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_arg)
    return nemo_asr.models.ASRModel.from_pretrained(model_arg)


def main():
    parser = argparse.ArgumentParser(description="Audit Parakeet/FastConformer-TDT architecture and config.")
    parser.add_argument("--model", required=True, help="Path to .nemo or HF model name")
    parser.add_argument("--out", default="audit_model_arch.json", help="Output JSON path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for probing")
    parser.add_argument("--time-steps", type=int, default=800, help="Probe input length for subsampling inference")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    model = load_model(args.model)
    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    encoder = model.encoder
    decoder = model.decoder
    joint = getattr(model, "joint", None)

    encoder_cfg = _to_container(getattr(model.cfg, "encoder", None))
    decoder_cfg = _to_container(getattr(model.cfg, "decoder", None))

    conv_stats = _collect_conv_stats(encoder)
    conv_kernel_sizes = sorted(
        {tuple(c["kernel_size"]) for c in conv_stats["all"] if c["kernel_size"] is not None}
    )

    attn_context = _pick_first_value(
        encoder_cfg,
        [
            "attn_context_size",
            "self_attention_model",
            "attention_type",
            "attn_type",
            "context_size",
        ],
    )

    subsampling_factor_cfg = _pick_first_value(
        encoder_cfg,
        [
            "subsampling_factor",
            "subsampling",
            "subsample_factor",
        ],
    )

    conv_kernel_cfg = _pick_first_value(
        encoder_cfg,
        [
            "conv_kernel_size",
            "conv_kernel",
            "conv_kernel_sizes",
            "kernel_size",
        ],
    )

    subsampling_channels = _pick_first_value(
        encoder_cfg,
        [
            "subsampling_channels",
            "subsample_channels",
            "conv_channels",
            "subsampling_conv_channels",
        ],
    )

    duration_values = None
    try:
        loss = getattr(joint, "loss", None) if joint is not None else None
        inner = getattr(loss, "_loss", None)
        dv = getattr(inner, "durations", None)
        if dv is not None:
            duration_values = list(dv)
    except Exception:
        duration_values = None

    duration_modules = []
    if joint is not None:
        for name, m in joint.named_modules():
            if "duration" in name.lower():
                duration_modules.append(name)

    subsampling_factor_inferred = None
    try:
        n_mels = int(model.cfg.preprocessor.get("features", 128))
        subsampling_factor_inferred = _infer_subsampling_factor(
            encoder, n_mels, device, args.time_steps
        )
    except Exception:
        subsampling_factor_inferred = None

    pos_emb_max_len = getattr(encoder, "pos_emb_max_len", None)
    streaming_cfg = getattr(encoder, "streaming_cfg", None)
    last_channel_cache_size = getattr(streaming_cfg, "last_channel_cache_size", None) if streaming_cfg else None

    checks = []
    checks.append({
        "id": "fastconformer_subsampling_factor_8",
        "desc": "FastConformer subsampling factor should be 8x",
        "value": subsampling_factor_cfg if subsampling_factor_cfg is not None else subsampling_factor_inferred,
        "expected": 8,
        "status": _status_from_expected(
            subsampling_factor_cfg if subsampling_factor_cfg is not None else subsampling_factor_inferred, 8
        ),
    })
    checks.append({
        "id": "fastconformer_kernel_size_9",
        "desc": "Conformer conv kernel size should be 9",
        "value": conv_kernel_cfg if conv_kernel_cfg is not None else conv_kernel_sizes,
        "expected": 9,
        "status": _status_from_expected(conv_kernel_cfg if conv_kernel_cfg is not None else 9 if (9,) in conv_kernel_sizes else None, 9),
    })
    checks.append({
        "id": "fastconformer_subsampling_channels_256",
        "desc": "Subsampling conv channels should be 256",
        "value": subsampling_channels,
        "expected": 256,
        "status": _status_from_expected(subsampling_channels, 256),
    })
    checks.append({
        "id": "tdt_duration_head_present",
        "desc": "TDT duration head/values should be present",
        "value": duration_values if duration_values is not None else duration_modules,
        "expected": "present",
        "status": "pass" if duration_values or duration_modules else "warn",
    })
    checks.append({
        "id": "cache_aware_streaming_api",
        "desc": "Encoder exposes cache-aware streaming hooks",
        "value": {
            "cache_aware_stream_step": hasattr(encoder, "cache_aware_stream_step"),
            "get_initial_cache_state": hasattr(encoder, "get_initial_cache_state"),
            "setup_streaming_params": hasattr(encoder, "setup_streaming_params"),
            "streaming_post_process": hasattr(encoder, "streaming_post_process"),
        },
        "expected": "all_true",
        "status": "pass" if all(
            hasattr(encoder, name) for name in [
                "cache_aware_stream_step",
                "get_initial_cache_state",
                "setup_streaming_params",
                "streaming_post_process",
            ]
        ) else "warn",
    })
    if pos_emb_max_len is not None and last_channel_cache_size is not None:
        status = "pass" if last_channel_cache_size <= pos_emb_max_len else "warn"
    else:
        status = "warn"
    checks.append({
        "id": "pos_emb_vs_cache_size",
        "desc": "last_channel_cache_size should not exceed pos_emb_max_len",
        "value": {
            "pos_emb_max_len": pos_emb_max_len,
            "last_channel_cache_size": last_channel_cache_size,
        },
        "expected": "cache_size <= pos_emb_max_len",
        "status": status,
    })

    report = {
        "model": {
            "source": args.model,
            "class": model.__class__.__name__,
        },
        "versions": {
            "torch": torch.__version__,
            "nemo": getattr(nemo_asr, "__version__", None),
        },
        "encoder": {
            "class": encoder.__class__.__name__,
            "cfg": encoder_cfg,
            "pos_emb_max_len": pos_emb_max_len,
            "streaming_cfg": _to_container(streaming_cfg),
            "conv_kernel_sizes": conv_kernel_sizes,
            "depthwise_conv_count": len(conv_stats["depthwise"]),
            "subsampling_conv_count": len(conv_stats["subsampling"]),
            "subsampling_factor_inferred": subsampling_factor_inferred,
            "attention_context": attn_context,
        },
        "decoder": {
            "class": decoder.__class__.__name__,
            "cfg": decoder_cfg,
        },
        "joint": {
            "class": joint.__class__.__name__ if joint is not None else None,
            "num_classes_with_blank": int(getattr(joint, "num_classes_with_blank", 0)) if joint is not None else None,
            "duration_values": duration_values,
            "duration_modules": duration_modules,
        },
        "checks": checks,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote audit report: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
