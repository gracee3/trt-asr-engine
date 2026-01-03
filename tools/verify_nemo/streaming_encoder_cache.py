#!/usr/bin/env python3
import argparse
import dataclasses
import inspect
import os
import sys

import torch
import nemo.collections.asr as nemo_asr


def _print_sig(label, fn):
    try:
        print(f"{label} signature: {inspect.signature(fn)}")
    except Exception as exc:
        print(f"{label} signature: <unavailable> ({exc})")


def _describe_value(prefix, value, *, max_list=6):
    if torch.is_tensor(value):
        shape = list(value.shape)
        msg = f"{prefix}: Tensor shape={shape} dtype={value.dtype} device={value.device}"
        if value.numel() <= 8:
            try:
                msg += f" value={value.detach().flatten().tolist()}"
            except Exception:
                pass
        print(msg)
        return
    if isinstance(value, (list, tuple)):
        print(f"{prefix}: {type(value).__name__} len={len(value)}")
        for idx, item in enumerate(value[:max_list]):
            _describe_value(f"{prefix}[{idx}]", item, max_list=max_list)
        if len(value) > max_list:
            print(f"{prefix}[...]: <truncated {len(value) - max_list} items>")
        return
    print(f"{prefix}: {type(value).__name__} value={value}")


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
    # Fallback: pass explicit chunk/shift when signature expects them.
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


def _call_cache_aware_stream_step(encoder, x, x_len, cache_state):
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

    # Fallback to positional, assuming standard ordering.
    if len(params) >= 5:
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

    # Best-effort by name.
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


def _assert_cache_len_ok(cache_len):
    if not torch.is_tensor(cache_len):
        raise RuntimeError(f"cache_last_channel_len is not a tensor: {type(cache_len)}")
    if torch.any(cache_len < 0):
        raise RuntimeError(f"cache_last_channel_len went negative: {cache_len}")


def load_model(model_arg):
    if os.path.exists(model_arg):
        try:
            return nemo_asr.models.ASRModel.restore_from(model_arg)
        except Exception as exc:
            print(f"ASRModel.restore_from failed: {exc}")
            return nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_arg)
    return nemo_asr.models.ASRModel.from_pretrained(model_arg)


def main():
    parser = argparse.ArgumentParser(
        description="Sanity-check encoder cache-aware streaming loop in PyTorch."
    )
    parser.add_argument("--model", required=True, help="Path to .nemo or HF model name")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--chunk-len", type=int, default=0, help="Feature frames per chunk")
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--chunk-size", type=str, default="", help="Override streaming_cfg.chunk_size (comma-separated)")
    parser.add_argument("--shift-size", type=str, default="", help="Override streaming_cfg.shift_size (comma-separated)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep-chunk-lens", type=str, default="", help="Comma-separated chunk lengths to test (chunk 0 only)")
    parser.add_argument("--no-assert-cache-len", action="store_true", help="Skip cache_len non-negative assertion")
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

    encoder = model.encoder
    _print_sig("encoder.cache_aware_stream_step", encoder.cache_aware_stream_step)
    if hasattr(encoder, "streaming_post_process"):
        _print_sig("encoder.streaming_post_process", encoder.streaming_post_process)
    if hasattr(encoder, "get_initial_cache_state"):
        _print_sig("encoder.get_initial_cache_state", encoder.get_initial_cache_state)

    if hasattr(encoder, "streaming_cfg"):
        cfg = encoder.streaming_cfg
        print(f"Original streaming_cfg: {cfg}")
        if dataclasses.is_dataclass(cfg):
            updates = {"last_channel_cache_size": args.cache_size}
            if args.chunk_size:
                updates["chunk_size"] = _parse_int_list(args.chunk_size)
            if args.shift_size:
                updates["shift_size"] = _parse_int_list(args.shift_size)
            encoder.streaming_cfg = dataclasses.replace(cfg, **updates)
        else:
            if hasattr(cfg, "last_channel_cache_size"):
                cfg.last_channel_cache_size = args.cache_size
            if args.chunk_size and hasattr(cfg, "chunk_size"):
                cfg.chunk_size = _parse_int_list(args.chunk_size)
            if args.shift_size and hasattr(cfg, "shift_size"):
                cfg.shift_size = _parse_int_list(args.shift_size)
        print(f"Updated streaming_cfg: {encoder.streaming_cfg}")
    else:
        print("WARN: encoder has no streaming_cfg; cache-aware streaming may be unavailable.")

    _call_setup_streaming_params(encoder)
    # setup_streaming_params resets streaming_cfg; reapply the cache size override.
    if hasattr(encoder, "streaming_cfg") and hasattr(encoder.streaming_cfg, "last_channel_cache_size"):
        encoder.streaming_cfg.last_channel_cache_size = args.cache_size
        print(f"Post-setup streaming_cfg: {encoder.streaming_cfg}")

    feature_dim = int(model.cfg.preprocessor.features)
    default_chunk_len = args.chunk_len or _select_chunk_len(
        getattr(encoder, "streaming_cfg", None), fallback=584
    )
    print(f"Using default chunk_len={default_chunk_len} feature_dim={feature_dim}")

    def _print_streaming_cfg_summary():
        cfg = getattr(encoder, "streaming_cfg", None)
        if cfg is None:
            return
        print("streaming_cfg summary:")
        print(f"  chunk_size={getattr(cfg, 'chunk_size', None)} shift_size={getattr(cfg, 'shift_size', None)}")
        print(f"  cache_drop_size={getattr(cfg, 'cache_drop_size', None)} valid_out_len={getattr(cfg, 'valid_out_len', None)}")
        print(f"  pre_encode_cache_size={getattr(cfg, 'pre_encode_cache_size', None)} drop_extra_pre_encoded={getattr(cfg, 'drop_extra_pre_encoded', None)}")
        print(f"  last_channel_cache_size={getattr(cfg, 'last_channel_cache_size', None)}")

    def _scalar_int(t):
        if not torch.is_tensor(t) or t.numel() != 1:
            return None
        return int(t.detach().item())

    def _log_chunk_summary(tag, step_out, post_out, chunk_len):
        cfg = getattr(encoder, "streaming_cfg", None)
        cache_drop_size = getattr(cfg, "cache_drop_size", None) if cfg is not None else None
        valid_out_len = getattr(cfg, "valid_out_len", None) if cfg is not None else None
        shift_size = getattr(cfg, "shift_size", None) if cfg is not None else None
        pre_encode_cache_size = getattr(cfg, "pre_encode_cache_size", None) if cfg is not None else None
        drop_extra_pre_encoded = getattr(cfg, "drop_extra_pre_encoded", None) if cfg is not None else None

        enc_len_pre = _scalar_int(step_out[1]) if isinstance(step_out, (tuple, list)) and len(step_out) > 1 else None
        cache_len_pre = _scalar_int(step_out[4]) if isinstance(step_out, (tuple, list)) and len(step_out) > 4 else None
        enc_len_post = _scalar_int(post_out[1]) if isinstance(post_out, (tuple, list)) and len(post_out) > 1 else None
        cache_len_post = _scalar_int(post_out[4]) if isinstance(post_out, (tuple, list)) and len(post_out) > 4 else None

        def _diff(a, b):
            if a is None or b is None:
                return None
            return a - b

        diff_drop = _diff(enc_len_pre, cache_drop_size)
        diff_drop_valid = _diff(enc_len_pre, (cache_drop_size + valid_out_len) if cache_drop_size is not None and valid_out_len is not None else None)
        print(f"{tag} summary:")
        print(f"  input_len={chunk_len} enc_len_pre={enc_len_pre} enc_len_post={enc_len_post}")
        print(f"  cache_len_pre={cache_len_pre} cache_len_post={cache_len_post}")
        print(f"  cache_drop_size={cache_drop_size} valid_out_len={valid_out_len} shift_size={shift_size}")
        print(f"  pre_encode_cache_size={pre_encode_cache_size} drop_extra_pre_encoded={drop_extra_pre_encoded}")
        print(f"  enc_len_pre - cache_drop_size={diff_drop} enc_len_pre - (cache_drop+valid)={diff_drop_valid}")

    def _run_sequence(chunk_len, num_chunks):
        x = torch.randn(1, feature_dim, chunk_len, device=device)
        x_len = torch.tensor([chunk_len], dtype=torch.int64, device=device)
        cache_state = _call_get_initial_cache_state(encoder, batch_size=1)
        cache_state = tuple(
            t.to(device) if torch.is_tensor(t) else t for t in cache_state
        )
        _describe_value("initial_cache_state", cache_state)

        with torch.no_grad():
            for idx in range(num_chunks):
                print(f"\n=== Chunk {idx} ===")
                step_out = _call_cache_aware_stream_step(encoder, x, x_len, cache_state)
                _describe_value("cache_aware_stream_step out", step_out)

                post_out = _call_streaming_post_process(encoder, step_out)
                if post_out is not step_out:
                    _describe_value("streaming_post_process out", post_out)

                _log_chunk_summary("chunk", step_out, post_out, chunk_len)
                cache_state = _extract_cache_state(post_out)
                _describe_value("next_cache_state", cache_state)
                if not args.no_assert_cache_len:
                    _assert_cache_len_ok(cache_state[2])

    _print_streaming_cfg_summary()

    if args.sweep_chunk_lens:
        sweep = _parse_int_list(args.sweep_chunk_lens)
        print(f"Running chunk-length sweep: {sweep}")
        for length in sweep:
            print(f"\n=== Sweep chunk_len={length} ===")
            try:
                _run_sequence(length, num_chunks=1)
            except Exception as exc:
                print(f"Sweep chunk_len={length} failed: {exc}")
        return

    _run_sequence(default_chunk_len, num_chunks=args.num_chunks)

    print("\n✅ Encoder cache-aware streaming loop completed without shape errors.")


if __name__ == "__main__":
    sys.exit(main())
