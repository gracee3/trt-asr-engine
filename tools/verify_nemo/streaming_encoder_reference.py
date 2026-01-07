#!/usr/bin/env python3
"""
Generate PyTorch reference JSONL with full tensor data for ONNX/ORT parity testing.
Based on streaming_encoder_cache.py but saves inputs/outputs as base64-encoded arrays.
"""
import argparse
import base64
import dataclasses
import inspect
import json
import os
import sys
import time

import numpy as np
import torch
import nemo.collections.asr as nemo_asr


# ----------------------------
# Utilities from streaming_encoder_cache.py
# ----------------------------
def _print_sig(label, fn):
    try:
        print(f"{label} signature: {inspect.signature(fn)}")
    except Exception as exc:
        print(f"{label} signature: <unavailable> ({exc})")


def _parse_int_list(value):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _select_chunk_len(streaming_cfg, fallback):
    if streaming_cfg is None:
        return fallback
    chunk_size = getattr(streaming_cfg, "chunk_size", None)
    if chunk_size is None:
        return fallback
    if isinstance(chunk_size, (list, tuple)):
        return int(max(chunk_size))
    return int(chunk_size)

def _infer_chunk_shift_steps(encoder, streaming_cfg):
    if streaming_cfg is None:
        return None, None
    chunk = getattr(streaming_cfg, "chunk_size", None)
    shift = getattr(streaming_cfg, "shift_size", None)
    cache_drop = getattr(streaming_cfg, "cache_drop_size", None)
    subsampling = getattr(encoder, "subsampling_factor", None) or 1

    if hasattr(encoder, "pre_encode") and hasattr(encoder.pre_encode, "get_sampling_frames"):
        sampling_frames = encoder.pre_encode.get_sampling_frames()
    else:
        sampling_frames = 0

    chunk_max = max(chunk) if isinstance(chunk, (list, tuple)) else chunk
    sampling_max = max(sampling_frames) if isinstance(sampling_frames, (list, tuple)) else sampling_frames
    if chunk_max is None or sampling_max is None or subsampling <= 0:
        return None, None

    try:
        lookahead_steps = int(round((chunk_max - sampling_max) / subsampling))
        chunk_steps = lookahead_steps + 1
    except Exception:
        return None, None

    shift_steps = None
    if cache_drop is not None:
        shift_steps = int(chunk_steps - cache_drop)
        if shift_steps < 1:
            shift_steps = 1
    elif shift is not None:
        shift_max = max(shift) if isinstance(shift, (list, tuple)) else shift
        if shift_max is not None and sampling_max is not None and subsampling > 0:
            try:
                lookahead_shift = int(round((shift_max - sampling_max) / subsampling))
                shift_steps = lookahead_shift + 1
            except Exception:
                shift_steps = None
    return chunk_steps, shift_steps


def _call_setup_streaming_params(encoder):
    if not hasattr(encoder, "setup_streaming_params"):
        return
    setup_fn = encoder.setup_streaming_params
    sig = inspect.signature(setup_fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if not params:
        setup_fn()
        return
    names = {p.name for p in params}
    if "streaming_cfg" in names:
        setup_fn(streaming_cfg=encoder.streaming_cfg)
        return
    if "cfg" in names:
        setup_fn(cfg=encoder.streaming_cfg)
        return
    if "chunk_size" in names or "shift_size" in names:
        cfg = getattr(encoder, "streaming_cfg", None)
        kwargs = {}
        inferred_chunk, inferred_shift = _infer_chunk_shift_steps(encoder, cfg)
        if "chunk_size" in names:
            value = inferred_chunk
            if value is None:
                value = getattr(cfg, "chunk_size", None) if cfg is not None else None
                if isinstance(value, (list, tuple)):
                    value = int(max(value)) if value else None
            kwargs["chunk_size"] = value
        if "shift_size" in names:
            value = inferred_shift
            if value is None:
                value = getattr(cfg, "shift_size", None) if cfg is not None else None
                if isinstance(value, (list, tuple)):
                    value = int(max(value)) if value else None
            kwargs["shift_size"] = value
        if "left_chunks" in names:
            kwargs["left_chunks"] = getattr(cfg, "left_chunks", None) if cfg is not None else None
        if "att_context_size" in names:
            kwargs["att_context_size"] = getattr(cfg, "att_context_size", None) if cfg is not None else None
        if "max_context" in names:
            kwargs["max_context"] = getattr(cfg, "max_context", None) if cfg is not None else None
        setup_fn(**kwargs)
        return
    raise RuntimeError(f"Unsupported setup_streaming_params signature: {sig}")


def _call_get_initial_cache_state(encoder, batch_size):
    if not hasattr(encoder, "get_initial_cache_state"):
        raise RuntimeError("Encoder does not expose get_initial_cache_state()")
    fn = encoder.get_initial_cache_state
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if not params:
        return fn()
    if any(p.name == "batch_size" for p in params):
        return fn(batch_size=batch_size)
    return fn(batch_size)


def _call_cache_aware_stream_step(encoder, x, x_len, cache_state, keep_all_outputs=None):
    cache_last_channel, cache_last_time, cache_last_channel_len = cache_state
    fn = encoder.cache_aware_stream_step
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    names = [p.name for p in params]

    def _pick(candidates):
        for name in candidates:
            if name in names:
                return name
        return None

    audio_name = _pick(["x", "audio_signal", "features", "input_signal"])
    length_name = _pick(["x_len", "length", "input_length", "audio_signal_length", "feature_length"])
    cache_channel_name = _pick(["cache_last_channel"])
    cache_time_name = _pick(["cache_last_time"])
    cache_len_name = _pick(["cache_last_channel_len"])

    if audio_name or length_name:
        kwargs = {}
        if audio_name:
            kwargs[audio_name] = x
        if length_name:
            kwargs[length_name] = x_len
        if cache_channel_name:
            kwargs[cache_channel_name] = cache_last_channel
        if cache_time_name:
            kwargs[cache_time_name] = cache_last_time
        if cache_len_name:
            kwargs[cache_len_name] = cache_last_channel_len
        if keep_all_outputs is not None and "keep_all_outputs" in names:
            kwargs["keep_all_outputs"] = keep_all_outputs

        missing = [
            p.name
            for p in params
            if p.default is inspect._empty and p.name not in kwargs
        ]
        if missing:
            raise RuntimeError(
                f"cache_aware_stream_step missing required args {missing}; "
                f"signature={sig}"
            )
        return fn(**kwargs)

    # Fallback to positional
    if len(params) >= 5:
        if keep_all_outputs is not None and "keep_all_outputs" in names:
            return fn(x, x_len, cache_last_channel, cache_last_time, cache_last_channel_len, keep_all_outputs=keep_all_outputs)
        return fn(x, x_len, cache_last_channel, cache_last_time, cache_last_channel_len)
    if len(params) == 4:
        return fn(x, x_len, cache_last_channel, cache_last_time)
    if len(params) == 3:
        return fn(x, x_len, cache_last_channel)
    raise RuntimeError(f"Unsupported cache_aware_stream_step signature: {sig}")


def _call_streaming_post_process(encoder, step_out):
    if not hasattr(encoder, "streaming_post_process"):
        return step_out
    fn = encoder.streaming_post_process
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if not params:
        return fn()

    step_tuple = step_out if isinstance(step_out, (tuple, list)) else (step_out,)
    if params and params[0].name == "rets":
        return fn(step_tuple)
    if len(params) == 1:
        return fn(step_tuple if len(step_tuple) > 1 else step_tuple[0])
    if len(params) == len(step_tuple):
        return fn(*step_tuple)

    # Best-effort by name
    names = [p.name for p in params]
    positional = {
        0: ["encoder_output", "encoder_out", "x", "encoded", "encoded_output"],
        1: ["encoded_length", "encoder_output_length", "x_len", "length", "encoded_len"],
        2: ["cache_last_channel"],
        3: ["cache_last_time"],
        4: ["cache_last_channel_len"],
    }
    value_by_name = {}
    for idx, candidates in positional.items():
        if idx >= len(step_tuple):
            continue
        for name in candidates:
            value_by_name[name] = step_tuple[idx]

    kwargs = {}
    for name in names:
        if name in value_by_name:
            kwargs[name] = value_by_name[name]

    missing = [
        p.name
        for p in params
        if p.default is inspect._empty and p.name not in kwargs
    ]
    if missing:
        if len(step_tuple) >= len(params):
            return fn(*step_tuple[: len(params)])
        raise RuntimeError(
            f"streaming_post_process missing required args {missing}; "
            f"signature={sig}"
        )
    return fn(**kwargs)


def _extract_cache_state(step_out):
    if not isinstance(step_out, (tuple, list)):
        raise RuntimeError("Expected tuple output to extract cache state.")
    if len(step_out) < 5:
        raise RuntimeError(
            f"Expected >=5 outputs for cache state, got {len(step_out)}"
        )
    return step_out[2], step_out[3], step_out[4]

def _normalize_streaming_outputs(step_out):
    if not isinstance(step_out, (tuple, list)) or len(step_out) < 5:
        raise RuntimeError(f"Expected >=5 streaming outputs, got {type(step_out)} len={len(step_out) if isinstance(step_out, (tuple, list)) else 'n/a'}")
    return step_out[0], step_out[1], step_out[2], step_out[3], step_out[4]

def _min_streaming_chunk_size(cfg):
    if cfg is None:
        return None
    chunk_size = getattr(cfg, "chunk_size", None)
    if chunk_size is None:
        return None
    if isinstance(chunk_size, (list, tuple)):
        return int(min(chunk_size)) if chunk_size else None
    return int(chunk_size)

def _compute_pre_encode_len(encoder, feature_dim: int, chunk_len: int, device):
    if not hasattr(encoder, "pre_encode"):
        return None
    dummy = torch.randn(1, feature_dim, chunk_len, device=device)
    length = torch.tensor([chunk_len], dtype=torch.int64, device=device)
    with torch.no_grad():
        if isinstance(encoder.pre_encode, torch.nn.Linear):
            length_out = length
        else:
            _, length_out = encoder.pre_encode(x=dummy.transpose(1, 2), lengths=length)
    try:
        return int(length_out.item())
    except Exception:
        return None

def _clamp_cache_drop_size(encoder, cfg, cache_drop_size, feature_dim, device):
    if cache_drop_size is None or cfg is None:
        return cache_drop_size
    chunk_min = _min_streaming_chunk_size(cfg)
    if chunk_min is None or chunk_min <= 0:
        return cache_drop_size
    pre_len = _compute_pre_encode_len(encoder, feature_dim, chunk_min, device)
    if pre_len is None:
        return cache_drop_size
    drop_extra = getattr(cfg, "drop_extra_pre_encoded", 0) or 0
    min_len = max(pre_len - int(drop_extra), 0)
    if cache_drop_size > min_len:
        print(
            f"WARN: cache_drop_size={cache_drop_size} exceeds pre-encoded length after drop_extra_pre_encoded "
            f"({min_len}); clamping to {min_len} to avoid negative cache_len."
        )
        return min_len
    return cache_drop_size

def pad_or_trunc_along_axis(x, axis, target, pad_side="right"):
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

def pad_or_trunc_tensor(x, axis, target):
    if not torch.is_tensor(x):
        return x
    cur = x.size(axis)
    if cur == target:
        return x
    if cur > target:
        return x.narrow(axis, 0, target)
    new_shape = list(x.shape)
    new_shape[axis] = target
    out = x.new_zeros(new_shape)
    out.narrow(axis, 0, cur).copy_(x)
    return out

def normalize_cache_state(cache_state, cache_size, time_ctx):
    cache_last_channel, cache_last_time, cache_last_channel_len = cache_state
    cache_last_channel = pad_or_trunc_tensor(cache_last_channel, axis=2, target=cache_size)
    cache_last_time = pad_or_trunc_tensor(cache_last_time, axis=3, target=time_ctx)
    return cache_last_channel, cache_last_time, cache_last_channel_len


def load_model(model_arg):
    if os.path.exists(model_arg):
        try:
            return nemo_asr.models.ASRModel.restore_from(model_arg)
        except Exception as exc:
            print(f"ASRModel.restore_from failed: {exc}")
            return nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_arg)
    return nemo_asr.models.ASRModel.from_pretrained(model_arg)


# ----------------------------
# Tensor encoding utilities
# ----------------------------
def encode_tensor(t):
    """Encode a torch/numpy tensor to a JSONL-compatible dict with base64 data."""
    if torch.is_tensor(t):
        arr = t.detach().cpu().numpy()
    elif isinstance(t, np.ndarray):
        arr = t
    else:
        raise ValueError(f"Unsupported tensor type: {type(t)}")

    # Convert to contiguous array
    arr = np.ascontiguousarray(arr)

    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference JSONL with tensor data for ORT parity testing."
    )
    parser.add_argument("--model", required=True, help="Path to .nemo or HF model name")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--time-ctx", type=int, default=4, help="Cache time context (conv kernel-1)")
    parser.add_argument("--cache-drop-size", type=int, default=-1, help="Override streaming_cfg.cache_drop_size (>=0)")
    parser.add_argument("--chunk-len", type=int, default=0, help="Feature frames per chunk")
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--chunk-size", type=str, default="", help="Override streaming_cfg.chunk_size (comma-separated)")
    parser.add_argument("--shift-size", type=str, default="", help="Override streaming_cfg.shift_size (comma-separated)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-setup-streaming-params", action="store_true", help="Skip encoder.setup_streaming_params call")
    parser.add_argument("--force-postprocess", action="store_true", help="Apply streaming_post_process after cache_aware_stream_step")
    parser.add_argument("--use-streaming-cfg-schedule", action="store_true", help="Use streaming_cfg chunk/shift/pre-encode schedule")
    parser.add_argument("--jsonl-out", type=str, required=True, help="Write per-chunk reference JSONL with tensor data")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    if args.seed:
        torch.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    model = load_model(args.model)
    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    feature_dim = int(model.cfg.preprocessor.features)
    encoder = model.encoder
    use_export_interface = hasattr(encoder, "forward_for_export")

    def _apply_cache_drop_size(encoder, cache_drop_size):
        if not hasattr(encoder, "streaming_cfg") or cache_drop_size is None:
            return
        try:
            if hasattr(encoder.streaming_cfg, "cache_drop_size"):
                encoder.streaming_cfg.cache_drop_size = cache_drop_size
            if hasattr(encoder, "layers"):
                for m in encoder.layers.modules():
                    if hasattr(m, "cache_drop_size"):
                        m.cache_drop_size = cache_drop_size
        except Exception as exc:
            print(f"WARN: failed to apply cache_drop_size override: {exc}")
    _print_sig("encoder.cache_aware_stream_step", encoder.cache_aware_stream_step)
    if hasattr(encoder, "streaming_post_process"):
        _print_sig("encoder.streaming_post_process", encoder.streaming_post_process)
    if hasattr(encoder, "get_initial_cache_state"):
        _print_sig("encoder.get_initial_cache_state", encoder.get_initial_cache_state)

    cache_drop_size_value = None
    if hasattr(encoder, "streaming_cfg"):
        cfg = encoder.streaming_cfg
        print(f"Original streaming_cfg: {cfg}")
        if dataclasses.is_dataclass(cfg):
            updates = {"last_channel_cache_size": args.cache_size}
            if args.chunk_size:
                updates["chunk_size"] = _parse_int_list(args.chunk_size)
            if args.shift_size:
                updates["shift_size"] = _parse_int_list(args.shift_size)
            if args.cache_drop_size >= 0:
                updates["cache_drop_size"] = int(args.cache_drop_size)
            encoder.streaming_cfg = dataclasses.replace(cfg, **updates)
        else:
            if hasattr(cfg, "last_channel_cache_size"):
                cfg.last_channel_cache_size = args.cache_size
            if args.chunk_size and hasattr(cfg, "chunk_size"):
                cfg.chunk_size = _parse_int_list(args.chunk_size)
            if args.shift_size and hasattr(cfg, "shift_size"):
                cfg.shift_size = _parse_int_list(args.shift_size)
            if args.cache_drop_size >= 0 and hasattr(cfg, "cache_drop_size"):
                cfg.cache_drop_size = int(args.cache_drop_size)
        if args.cache_drop_size >= 0:
            _apply_cache_drop_size(encoder, int(args.cache_drop_size))
        cache_drop_size_value = getattr(encoder.streaming_cfg, "cache_drop_size", None)
        safe_drop = _clamp_cache_drop_size(encoder, encoder.streaming_cfg, cache_drop_size_value, feature_dim, device)
        if safe_drop is not None and safe_drop != cache_drop_size_value:
            if dataclasses.is_dataclass(encoder.streaming_cfg):
                encoder.streaming_cfg = dataclasses.replace(encoder.streaming_cfg, cache_drop_size=safe_drop)
            else:
                if hasattr(encoder.streaming_cfg, "cache_drop_size"):
                    encoder.streaming_cfg.cache_drop_size = safe_drop
            _apply_cache_drop_size(encoder, int(safe_drop))
            cache_drop_size_value = safe_drop
        print(f"Updated streaming_cfg: {encoder.streaming_cfg}")
    else:
        print("WARN: encoder has no streaming_cfg; cache-aware streaming may be unavailable.")

    if not args.skip_setup_streaming_params:
        _call_setup_streaming_params(encoder)
        # setup_streaming_params resets streaming_cfg; reapply the cache size override
        if hasattr(encoder, "streaming_cfg") and hasattr(encoder.streaming_cfg, "last_channel_cache_size"):
            encoder.streaming_cfg.last_channel_cache_size = args.cache_size
        if cache_drop_size_value is not None and hasattr(encoder, "streaming_cfg"):
            if dataclasses.is_dataclass(encoder.streaming_cfg):
                encoder.streaming_cfg = dataclasses.replace(encoder.streaming_cfg, cache_drop_size=cache_drop_size_value)
            else:
                if hasattr(encoder.streaming_cfg, "cache_drop_size"):
                    encoder.streaming_cfg.cache_drop_size = cache_drop_size_value
            _apply_cache_drop_size(encoder, int(cache_drop_size_value))
        post_safe = _clamp_cache_drop_size(encoder, getattr(encoder, "streaming_cfg", None), cache_drop_size_value, feature_dim, device)
        if post_safe is not None and post_safe != cache_drop_size_value:
            if dataclasses.is_dataclass(encoder.streaming_cfg):
                encoder.streaming_cfg = dataclasses.replace(encoder.streaming_cfg, cache_drop_size=post_safe)
            else:
                if hasattr(encoder.streaming_cfg, "cache_drop_size"):
                    encoder.streaming_cfg.cache_drop_size = post_safe
            _apply_cache_drop_size(encoder, int(post_safe))
            cache_drop_size_value = post_safe
            _call_setup_streaming_params(encoder)
            if hasattr(encoder.streaming_cfg, "last_channel_cache_size"):
                encoder.streaming_cfg.last_channel_cache_size = args.cache_size
        print(f"Post-setup streaming_cfg: {encoder.streaming_cfg}")

    chunk_len = args.chunk_len or _select_chunk_len(
        getattr(encoder, "streaming_cfg", None), fallback=584
    )
    print(f"Using chunk_len={chunk_len} feature_dim={feature_dim}")

    cfg = getattr(encoder, "streaming_cfg", None)
    print(f"Streaming config: chunk_size={getattr(cfg, 'chunk_size', None)} shift_size={getattr(cfg, 'shift_size', None)}")
    print(f"  cache_drop_size={getattr(cfg, 'cache_drop_size', None)} valid_out_len={getattr(cfg, 'valid_out_len', None)}")

    def _normalize_pair(value, fallback):
        if value is None:
            return [fallback, fallback]
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                return [value[0], value[1]]
            if len(value) == 1:
                return [value[0], value[0]]
            return [fallback, fallback]
        return [value, value]

    def _build_schedule(num_chunks, chunk_len_default):
        cfg = getattr(encoder, "streaming_cfg", None)
        chunk_sizes = _normalize_pair(getattr(cfg, "chunk_size", None) if cfg else None, chunk_len_default)
        shift_sizes = _normalize_pair(getattr(cfg, "shift_size", None) if cfg else None, chunk_sizes[1])
        pre_encode_sizes = _normalize_pair(getattr(cfg, "pre_encode_cache_size", None) if cfg else None, 0)

        schedule = []
        start = 0
        for idx in range(num_chunks):
            regime = 0 if idx == 0 else 1
            chunk_size = int(chunk_sizes[regime])
            shift_size = int(shift_sizes[regime])
            pre_encode = int(pre_encode_sizes[regime])
            slice_start = max(0, start - pre_encode)
            slice_end = start + chunk_size
            schedule.append(
                {
                    "idx": idx,
                    "start": start,
                    "chunk_size": chunk_size,
                    "shift_size": shift_size,
                    "pre_encode": pre_encode,
                    "slice_start": slice_start,
                    "slice_end": slice_end,
                }
            )
            start += shift_size
        total_frames = max(item["slice_end"] for item in schedule) if schedule else chunk_len_default
        return schedule, total_frames

    # Initialize cache state
    cache_state = _call_get_initial_cache_state(encoder, batch_size=1)
    cache_state = tuple(
        t.to(device) if torch.is_tensor(t) else t for t in cache_state
    )
    if use_export_interface:
        cache_state = (
            cache_state[0].transpose(0, 1).contiguous(),
            cache_state[1].transpose(0, 1).contiguous(),
            cache_state[2],
        )

    # Generate random audio sequence
    if args.use_streaming_cfg_schedule:
        schedule, total_frames = _build_schedule(args.num_chunks, chunk_len)
        feature_buffer = torch.randn(1, feature_dim, total_frames, device=device)
    else:
        schedule = None
        feature_buffer = torch.randn(1, feature_dim, chunk_len * args.num_chunks, device=device)

    with open(args.jsonl_out, "w", encoding="utf-8") as jsonl_handle:
        with torch.no_grad():
            for idx in range(args.num_chunks):
                print(f"\n=== Chunk {idx} ===")

                # Extract chunk
                schedule_item = None
                if schedule:
                    item = schedule[idx]
                    schedule_item = item
                    x = feature_buffer[:, :, item["slice_start"] : item["slice_end"]]
                    x_len_val = item["slice_end"] - item["slice_start"]
                else:
                    start_frame = idx * chunk_len
                    end_frame = (idx + 1) * chunk_len
                    x = feature_buffer[:, :, start_frame:end_frame]
                    x_len_val = chunk_len
                x_len = torch.tensor([x_len_val], dtype=torch.int64, device=device)

                # Save inputs
                cache_state = normalize_cache_state(cache_state, args.cache_size, args.time_ctx)
                inputs = {
                    "audio_signal": encode_tensor(x),
                    "length": encode_tensor(x_len),
                    "cache_last_channel": encode_tensor(cache_state[0]),
                    "cache_last_time": encode_tensor(cache_state[1]),
                    "cache_last_channel_len": encode_tensor(cache_state[2]),
                }

                # Run forward pass
                step_start = time.perf_counter()
                if use_export_interface:
                    step_out = encoder.forward_for_export(
                        x,
                        x_len,
                        cache_last_channel=cache_state[0],
                        cache_last_time=cache_state[1],
                        cache_last_channel_len=cache_state[2],
                    )
                else:
                    step_out = _call_cache_aware_stream_step(
                        encoder,
                        x,
                        x_len,
                        cache_state,
                        keep_all_outputs=False,
                    )
                step_ms = (time.perf_counter() - step_start) * 1000.0

                post_ms = 0.0
                if not use_export_interface and args.force_postprocess:
                    post_start = time.perf_counter()
                    step_out = _call_streaming_post_process(encoder, step_out)
                    post_ms = (time.perf_counter() - post_start) * 1000.0

                post_out = _normalize_streaming_outputs(step_out)

                # Extract outputs
                encoder_output = post_out[0]
                encoded_lengths = post_out[1]
                cache_last_channel_out = pad_or_trunc_tensor(post_out[2], axis=2, target=args.cache_size)
                cache_last_time_out = pad_or_trunc_tensor(post_out[3], axis=3, target=args.time_ctx)
                cache_last_channel_len_out = post_out[4]

                cache_len_val = int(cache_last_channel_len_out.item())
                if cache_len_val < 0:
                    raise RuntimeError(f"cache_last_channel_len_out went negative: {cache_len_val}")

                outputs = {
                    "encoder_output": encode_tensor(encoder_output),
                    "encoded_lengths": encode_tensor(encoded_lengths),
                    "cache_last_channel_out": encode_tensor(cache_last_channel_out),
                    "cache_last_time_out": encode_tensor(cache_last_time_out),
                    "cache_last_channel_len_out": encode_tensor(cache_last_channel_len_out),
                }

                # Build record
                metadata = {
                    "chunk_len": x_len_val,
                    "encoder_output_shape": list(encoder_output.shape),
                    "encoded_lengths_value": int(encoded_lengths.item()),
                    "cache_len_out_value": cache_len_val,
                    "timing_ms": {
                        "step": step_ms,
                        "postprocess": post_ms,
                        "total": step_ms + post_ms,
                    },
                }
                if schedule_item is not None:
                    metadata["schedule"] = {
                        "start": schedule_item["start"],
                        "chunk_size": schedule_item["chunk_size"],
                        "shift_size": schedule_item["shift_size"],
                        "pre_encode": schedule_item["pre_encode"],
                        "slice_start": schedule_item["slice_start"],
                        "slice_end": schedule_item["slice_end"],
                    }

                record = {
                    "chunk_idx": idx,
                    "inputs": inputs,
                    "outputs": outputs,
                    "metadata": metadata,
                }

                jsonl_handle.write(json.dumps(record) + "\n")
                jsonl_handle.flush()

                print(f"  encoder_output shape: {encoder_output.shape}")
                print(f"  encoded_lengths: {encoded_lengths.item()}")
                print(f"  cache_len_out: {cache_len_val}")
                print(f"  timing: {step_ms + post_ms:.2f}ms")

                # Update cache state for next iteration
                cache_state = (cache_last_channel_out, cache_last_time_out, cache_last_channel_len_out)

    print(f"\n✅ Reference JSONL written to: {args.jsonl_out}")
    print(f"   Total chunks: {args.num_chunks}")


if __name__ == "__main__":
    sys.exit(main())
