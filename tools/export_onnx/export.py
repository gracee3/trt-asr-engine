import argparse
import dataclasses
import os
import torch
import json
import time
import threading
import psutil
import onnx
import logging
from datetime import datetime
from contextlib import contextmanager
import inspect
import sys
import nemo.collections.asr as nemo_asr
from nemo.core.classes.common import typecheck

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _onnx_io_summary(path: str):
    """
    Return (inputs, outputs) as lists of dicts: {name: str, dims: [str|int]}.
    Uses load_external_data=False so it stays fast even for >2GB models.
    """
    m = onnx.load_model(path, load_external_data=False)

    def _dims(v):
        t = v.type.tensor_type
        out = []
        for d in t.shape.dim:
            if d.dim_param:
                out.append(d.dim_param)
            elif d.dim_value:
                out.append(int(d.dim_value))
            else:
                out.append("?")
        return out

    ins = [{"name": i.name, "dims": _dims(i)} for i in m.graph.input]
    outs = [{"name": o.name, "dims": _dims(o)} for o in m.graph.output]
    return ins, outs

def _parse_int_list(value: str):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]

def _select_streaming_chunk_len(streaming_cfg, fallback: int):
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

def _apply_cache_drop_size_override(encoder, cache_drop_size: int):
    try:
        if not hasattr(encoder, "layers"):
            return
        for m in encoder.layers.modules():
            if hasattr(m, "cache_drop_size"):
                m.cache_drop_size = cache_drop_size
    except Exception as exc:
        logger.warning(f"Failed to apply cache_drop_size override to encoder layers: {exc}")

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
        logger.warning(
            "cache_drop_size=%s exceeds pre-encoded length after drop_extra_pre_encoded (%s); "
            "clamping to %s to avoid negative cache_len.",
            cache_drop_size,
            min_len,
            min_len,
        )
        return min_len
    return cache_drop_size

def _maybe_update_streaming_cfg(encoder, cache_size, chunk_size, shift_size, cache_drop_size):
    cfg = getattr(encoder, "streaming_cfg", None)
    if cfg is None:
        logger.warning("Encoder has no streaming_cfg; cache-aware export may be unavailable.")
        return None
    updates = {}
    if cache_size is not None:
        updates["last_channel_cache_size"] = cache_size
    if chunk_size:
        updates["chunk_size"] = chunk_size
    if shift_size:
        updates["shift_size"] = shift_size
    if cache_drop_size is not None:
        updates["cache_drop_size"] = cache_drop_size
    if not updates:
        return cfg
    if dataclasses.is_dataclass(cfg):
        encoder.streaming_cfg = dataclasses.replace(cfg, **updates)
    else:
        for key, value in updates.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    if "cache_drop_size" in updates:
        _apply_cache_drop_size_override(encoder, updates["cache_drop_size"])
    logger.info(f"Updated encoder.streaming_cfg for streaming export: {encoder.streaming_cfg}")
    return getattr(encoder, "streaming_cfg", cfg)

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

def _call_get_initial_cache_state(encoder, batch_size: int):
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
    if len(params) == 1:
        return fn(step_tuple if len(step_tuple) > 1 else step_tuple[0])
    if len(params) == len(step_tuple):
        return fn(*step_tuple)
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

def _normalize_streaming_outputs(step_out):
    if not isinstance(step_out, (tuple, list)) or len(step_out) < 5:
        raise RuntimeError(f"Expected >=5 streaming outputs, got {type(step_out)} len={len(step_out) if isinstance(step_out, (tuple, list)) else 'n/a'}")
    return step_out[0], step_out[1], step_out[2], step_out[3], step_out[4]

class StreamingEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, use_post_process: bool):
        super().__init__()
        self.encoder = encoder
        self.use_post_process = use_post_process

    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
    ):
        if hasattr(self.encoder, "forward_for_export"):
            out = self.encoder.forward_for_export(
                audio_signal,
                length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )
            return _normalize_streaming_outputs(out)
        out = _call_cache_aware_stream_step(
            self.encoder,
            audio_signal,
            length,
            (cache_last_channel, cache_last_time, cache_last_channel_len),
            keep_all_outputs=False,
        )
        if self.use_post_process:
            out = _call_streaming_post_process(self.encoder, out)
        return _normalize_streaming_outputs(out)

class ExportHeartbeat(threading.Thread):
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()
        self.stage = "Initializing"
        self.start_time = time.time()

    def set_stage(self, stage):
        logger.info(f">>> STAGE CHANGE: {stage}")
        self.stage = stage

    def run(self):
        process = psutil.Process(os.getpid())
        while not self.stop_event.is_set():
            elapsed = time.time() - self.start_time
            rss = process.memory_info().rss / (1024 * 1024)  # MB
            print(f"[HEARTBEAT] Stage: {self.stage} | Elapsed: {elapsed:.1f}s | RSS: {rss:.1f} MB", flush=True)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

def validate_onnx(path):
    logger.info(f"Validating ONNX model at {path}...")
    try:
        # For large models (>2GB), check_model should be called with the path.
        # We also load w/ external_data disabled so we can print IO without pulling huge tensors.
        model = onnx.load_model(path, load_external_data=False)
        onnx.checker.check_model(path)

        def _shape_str(v):
            try:
                t = v.type.tensor_type
                dims = []
                for d in t.shape.dim:
                    if d.dim_param:
                        dims.append(d.dim_param)
                    elif d.dim_value:
                        dims.append(str(d.dim_value))
                    else:
                        dims.append("?")
                return f"{v.name}: [{', '.join(dims)}]"
            except Exception:
                return f"{getattr(v, 'name', '<unknown>')}: <unknown>"

        inputs = ", ".join(_shape_str(i) for i in model.graph.input)
        outputs = ", ".join(_shape_str(o) for o in model.graph.output)
        logger.info(f"ONNX IO | inputs: {inputs}")
        logger.info(f"ONNX IO | outputs: {outputs}")

        size_mb = os.path.getsize(path) / (1024 * 1024)
        data_path = path + ".data"
        if os.path.exists(data_path):
            data_size_gb = os.path.getsize(data_path) / (1024 * 1024 * 1024)
            logger.info(f"✅ ONNX VALIDATED: {path} (Data: {data_size_gb:.2f} GB)")
        else:
            logger.info(f"✅ ONNX VALIDATED: {path} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"❌ ONNX VALIDATION FAILED for {path}: {e}")
        return False

def _export_onnx_legacy(
    model: torch.nn.Module,
    args,
    path: str,
    *,
    input_names,
    output_names,
    dynamic_axes=None,
    opset_version=18,
):
    """
    Force legacy (non-dynamo / non-torch.export) ONNX export.

    torch 2.9 defaults to dynamo=True, which routes into torch.export/FakeTensor and can fail
    on NeMo control-flow / LSTM flattening. We explicitly disable it.
    """
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        logger.info("Using legacy exporter path: torch.onnx.export(dynamo=False) (no torch.export/dynamo)")
        torch.onnx.export(
            model,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=False,
            dynamo=False,
            fallback=False,
            external_data=True,
        )
    else:
        # Very old torch: only legacy exists.
        logger.info("Using legacy exporter path: torch.onnx.export (torch too old to support dynamo flag)")
        torch.onnx.export(
            model,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=False,
        )

@contextmanager
def _neutralize_dropout_attrs_for_export(root: torch.nn.Module, *, tag: str):
    """
    Export-only patch for NeMo modules that do `if self.dropout:` (TorchScript/torch.export fails
    because `Dropout` can't be cast to bool).

    We only patch attributes literally named "dropout" that are instances of nn.Dropout.
    We set them to None so the conditional becomes false and the forward remains valid.
    """
    patched = []
    try:
        for m in root.modules():
            if hasattr(m, "dropout"):
                d = getattr(m, "dropout")
                if isinstance(d, torch.nn.Dropout):
                    patched.append((m, "dropout", d))
                    setattr(m, "dropout", None)
        if patched:
            logger.info(f"[{tag}] Patched {len(patched)} modules: set `.dropout` (nn.Dropout) -> None for export")
        else:
            logger.info(f"[{tag}] No `.dropout` nn.Dropout attributes found to patch")
        yield patched
    finally:
        for m, name, orig in patched:
            setattr(m, name, orig)
        if patched:
            logger.info(f"[{tag}] Restored patched dropout attributes")

@contextmanager
def _disable_joint_fuse_loss_wer_for_export(joint: torch.nn.Module, *, tag: str):
    """
    Export-only patch: disable RNNTJoint fused loss/WER path, which forces transcript inputs.
    For inference ONNX export we want the pure joint network (encoder_out + predictor_out -> logits/durations).
    """
    orig = None
    try:
        orig = getattr(joint, "_fuse_loss_wer", None)
        if hasattr(joint, "set_fuse_loss_wer"):
            joint.set_fuse_loss_wer(False)
        elif orig is not None:
            setattr(joint, "_fuse_loss_wer", False)
        logger.info(f"[{tag}] fuse_loss_wer: {getattr(joint, 'fuse_loss_wer', '<unknown>')} (export override applied)")
        yield
    finally:
        try:
            if orig is not None:
                if hasattr(joint, "set_fuse_loss_wer"):
                    joint.set_fuse_loss_wer(bool(orig))
                else:
                    setattr(joint, "_fuse_loss_wer", orig)
                logger.info(f"[{tag}] fuse_loss_wer restored to {getattr(joint, 'fuse_loss_wer', '<unknown>')}")
        except Exception as e:
            logger.warning(f"[{tag}] Failed to restore fuse_loss_wer: {e}")

@contextmanager
def _force_joint_logits_for_export(joint: torch.nn.Module, *, tag: str):
    """
    Export-only patch: force RNNTJoint to emit raw logits (no log-softmax).
    This aligns with TDT's independently-normalized token/duration heads.
    """
    orig = None
    try:
        if hasattr(joint, "log_softmax"):
            orig = getattr(joint, "log_softmax", None)
            setattr(joint, "log_softmax", False)
            logger.info(f"[{tag}] log_softmax set to False for export (orig={orig})")
        yield
    finally:
        if orig is not None and hasattr(joint, "log_softmax"):
            setattr(joint, "log_softmax", orig)
            logger.info(f"[{tag}] log_softmax restored to {orig}")

class PredictorWrapper(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        # Best-effort infer LSTM state shapes for a stable ONNX signature.
        num_layers, hidden_size = 2, 640
        try:
            # NeMo RNNTDecoder commonly stores the prediction network here:
            # decoder.prediction['dec_rnn'].lstm
            pred = getattr(self.predictor, "prediction", None)
            if pred is not None and hasattr(pred, "__getitem__"):
                dec_rnn = pred["dec_rnn"]
                lstm = getattr(dec_rnn, "lstm", None)
                if lstm is not None:
                    num_layers = int(getattr(lstm, "num_layers", num_layers) or num_layers)
                    hidden_size = int(getattr(lstm, "hidden_size", hidden_size) or hidden_size)
        except Exception:
            pass
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, y, h, c):
        """
        Streaming-style predictor export:
        - inputs: token ids `y` and LSTM state tensors (h, c)
        - outputs: predictor embedding `g` and next state tensors (h', c')
        """
        # Prefer RNNTDecoder.predict() which is explicitly inference-oriented.
        if hasattr(self.predictor, "predict"):
            g, state = self.predictor.predict(y=y, state=[h, c], add_sos=False)
            # NeMo returns g as (B, U, D). For consistency with RNNTJoint input
            # (decoder_outputs expected as (B, D, U)), transpose here.
            g = g.transpose(1, 2)
            return g, state[0], state[1]

        # Fallback to forward(targets, target_length, states=...)
        # Note: target_length is derived from y.shape[1].
        target_length = torch.full((y.shape[0],), y.shape[1], dtype=torch.long, device=y.device)
        g, _, state = self.predictor(targets=y, target_length=target_length, states=[h, c])
        g = g.transpose(1, 2)
        return g, state[0], state[1]

class JointWrapper(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint
    def forward(self, encoder_output, predictor_output):
        """
        NeMo RNNT-TDT Joint can require lengths when `fuse_loss_wer` is enabled.
        We always provide them to keep export stable/deterministic.
        """
        out = self.joint(encoder_outputs=encoder_output, decoder_outputs=predictor_output)
        # In non-fused mode, NeMo RNNTJoint returns a single tensor: [B, T, U, V(+durations/...)].
        return out

def export_encoder(model, out_dir, device, dynamic=True):
    stage = "Exporting Encoder (Dynamic=" + str(dynamic) + ")"
    logger.info(f"Starting: {stage}")
    
    # 128 is features from model_meta we extracted previously
    n_mels = model.cfg.preprocessor.get('features', 128)
    
    # Use modest sizes for fixed export if requested
    time_steps = 64
    dummy_input = torch.randn(1, n_mels, time_steps).to(device)
    dummy_len = torch.LongTensor([time_steps]).to(device)
    
    path = os.path.join(out_dir, "encoder.onnx")
    
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'audio_signal': {0: 'batch', 2: 'time'},
            'encoder_output': {0: 'batch', 2: 'time'}
        }

    logger.info("Calling torch.onnx.export for Encoder...")
    _export_onnx_legacy(
        model.encoder,
        (dummy_input, dummy_len),
        path,
        input_names=['audio_signal', 'length'],
        output_names=['encoder_output', 'encoded_lengths'],
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    logger.info(f"Finished writing: {path}")
    validate_onnx(path)

def export_encoder_streaming(
    model,
    out_dir,
    device,
    dynamic=True,
    cache_size=256,
    cache_drop_size=None,
    chunk_size=None,
    shift_size=None,
    dummy_len=0,
    use_post_process=True,
    skip_setup=False,
):
    stage = "Exporting Encoder (Streaming Cache-Aware)"
    logger.info(f"Starting: {stage}")

    encoder = model.encoder
    if hasattr(encoder, "export_cache_support"):
        encoder.export_cache_support = True
        logger.info("encoder.export_cache_support set to True")
    else:
        logger.warning("encoder.export_cache_support not found; continuing without it")

    n_mels = model.cfg.preprocessor.get('features', 128)
    cfg = _maybe_update_streaming_cfg(encoder, cache_size, chunk_size, shift_size, cache_drop_size)
    if cache_drop_size is None and cfg is not None:
        cache_drop_size = getattr(cfg, "cache_drop_size", None)
    cache_drop_size = _clamp_cache_drop_size(encoder, cfg, cache_drop_size, n_mels, device)
    cfg = _maybe_update_streaming_cfg(encoder, cache_size, chunk_size, shift_size, cache_drop_size)
    if not skip_setup:
        _call_setup_streaming_params(encoder)
        # setup_streaming_params may reset streaming_cfg; reapply overrides if needed.
        cfg = _maybe_update_streaming_cfg(encoder, cache_size, chunk_size, shift_size, cache_drop_size)
        post_drop = _clamp_cache_drop_size(encoder, cfg, cache_drop_size, n_mels, device)
        if post_drop != cache_drop_size:
            cache_drop_size = post_drop
            cfg = _maybe_update_streaming_cfg(encoder, cache_size, chunk_size, shift_size, cache_drop_size)
            _call_setup_streaming_params(encoder)
    else:
        logger.info("Skipping encoder.setup_streaming_params per CLI flag")

    logger.info("Final encoder.streaming_cfg for streaming export: %s", getattr(encoder, "streaming_cfg", None))
    time_steps = dummy_len or _select_streaming_chunk_len(cfg, fallback=64)
    dummy_input = torch.randn(1, n_mels, time_steps).to(device)
    dummy_len_tensor = torch.LongTensor([time_steps]).to(device)

    cache_state = _call_get_initial_cache_state(encoder, batch_size=1)
    cache_state = tuple(
        t.to(device) if torch.is_tensor(t) else t for t in cache_state
    )
    if len(cache_state) < 3:
        raise RuntimeError(f"Expected 3 cache tensors, got {len(cache_state)}")
    use_batch_first_cache = hasattr(encoder, "forward_for_export")
    if use_batch_first_cache:
        cache_state = (
            cache_state[0].transpose(0, 1).contiguous(),
            cache_state[1].transpose(0, 1).contiguous(),
            cache_state[2],
        )

    wrapper = StreamingEncoderWrapper(encoder, use_post_process=use_post_process).to(device)

    path = os.path.join(out_dir, "encoder_streaming.onnx")
    dynamic_axes = None
    if dynamic:
        cache_batch_axis = 0 if use_batch_first_cache else 1
        dynamic_axes = {
            'audio_signal': {0: 'batch', 2: 'time'},
            'length': {0: 'batch'},
            'cache_last_channel': {cache_batch_axis: 'batch'},
            'cache_last_time': {cache_batch_axis: 'batch'},
            'cache_last_channel_len': {0: 'batch'},
            'encoder_output': {0: 'batch', 2: 'time'},
            'encoded_lengths': {0: 'batch'},
            'cache_last_channel_out': {cache_batch_axis: 'batch'},
            'cache_last_time_out': {cache_batch_axis: 'batch'},
            'cache_last_channel_len_out': {0: 'batch'},
        }

    logger.info("Calling torch.onnx.export for streaming encoder...")
    _export_onnx_legacy(
        wrapper,
        (dummy_input, dummy_len_tensor, cache_state[0], cache_state[1], cache_state[2]),
        path,
        input_names=[
            'audio_signal',
            'length',
            'cache_last_channel',
            'cache_last_time',
            'cache_last_channel_len',
        ],
        output_names=[
            'encoder_output',
            'encoded_lengths',
            'cache_last_channel_out',
            'cache_last_time_out',
            'cache_last_channel_len_out',
        ],
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    logger.info(f"Finished writing: {path}")
    validate_onnx(path)

def export_predictor(model, out_dir, device):
    stage = "Exporting Predictor"
    logger.info(f"Starting: {stage}")
    
    wrapper = PredictorWrapper(model.decoder).to(device)
    dummy_y = torch.zeros(1, 1, dtype=torch.long).to(device)
    dummy_h = torch.zeros(wrapper.num_layers, 1, wrapper.hidden_size, dtype=torch.float32).to(device)
    dummy_c = torch.zeros(wrapper.num_layers, 1, wrapper.hidden_size, dtype=torch.float32).to(device)
    
    path = os.path.join(out_dir, "predictor.onnx")
    
    with _neutralize_dropout_attrs_for_export(model.decoder, tag="predictor"):
        logger.info("Calling torch.onnx.export for Predictor (legacy, no torch.export/dynamo)...")
        _export_onnx_legacy(
            wrapper,
            (dummy_y, dummy_h, dummy_c),
            path,
            input_names=['y', 'h', 'c'],
            output_names=['g', 'h_out', 'c_out'],
            dynamic_axes={
                'y': {0: 'batch', 1: 'token_len'},
                'h': {1: 'batch'},
                'c': {1: 'batch'},
                'g': {0: 'batch', 2: 'token_len'},
                'h_out': {1: 'batch'},
                'c_out': {1: 'batch'},
            },
            opset_version=18,
        )
    logger.info(f"Finished writing: {path}")
    validate_onnx(path)

def export_joint(model, out_dir, device):
    stage = "Exporting Joint"
    logger.info(f"Starting: {stage}")
    
    wrapper = JointWrapper(model.joint).to(device)
    # Infer dims from the joint itself when possible (most reliable).
    h_enc = getattr(getattr(model.joint, "enc", None), "in_features", None) or 1024
    h_pred = getattr(getattr(model.joint, "pred", None), "in_features", None) or 640
    
    dummy_enc = torch.randn(1, h_enc, 1).to(device)
    dummy_pred = torch.randn(1, h_pred, 1).to(device)
    
    path = os.path.join(out_dir, "joint.onnx")
    
    with _neutralize_dropout_attrs_for_export(model.decoder, tag="joint/decoder"):
        with _neutralize_dropout_attrs_for_export(model.joint, tag="joint/joint"):
            with _disable_joint_fuse_loss_wer_for_export(model.joint, tag="joint"):
                with _force_joint_logits_for_export(model.joint, tag="joint"):
                    logger.info("Calling torch.onnx.export for Joint (legacy, no torch.export/dynamo)...")
                    _export_onnx_legacy(
                        wrapper,
                        (dummy_enc, dummy_pred),
                        path,
                        input_names=['encoder_output', 'predictor_output'],
                        output_names=['joint_output'],
                        dynamic_axes={
                            'encoder_output': {0: 'batch', 2: 'time'},
                            'predictor_output': {0: 'batch', 2: 'token_len'},
                            'joint_output': {0: 'batch', 1: 'time', 2: 'token_len'},
                        },
                        opset_version=18,
                    )
    logger.info(f"Finished writing: {path}")
    validate_onnx(path)

def _smoke_test_onnxruntime(predictor_path: str, joint_path: str):
    """
    Minimal ORT smoke test: run predictor once and joint once to catch exported-but-unusable graphs.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f"onnxruntime smoke test requested but dependencies missing: {e}")

    logger.info("Running ONNX Runtime smoke test (CPUExecutionProvider)...")
    sess_opt = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]

    # Predictor: y:int64 [1,1], h/c: float32 [2,1,640] -> g: float32 [1,640,1], h_out/c_out
    pred_sess = ort.InferenceSession(predictor_path, sess_options=sess_opt, providers=providers)
    pred_inputs = {i.name: i for i in pred_sess.get_inputs()}
    if not {"y", "h", "c"}.issubset(set(pred_inputs.keys())):
        raise RuntimeError(f"Predictor inputs unexpected: {list(pred_inputs.keys())}")
    y = np.zeros((1, 1), dtype=np.int64)
    h = np.zeros((2, 1, 640), dtype=np.float32)
    c = np.zeros((2, 1, 640), dtype=np.float32)
    g, h_out, c_out = pred_sess.run(None, {"y": y, "h": h, "c": c})
    logger.info(f"ORT predictor ok | g={g.shape} {g.dtype}, h_out={h_out.shape}, c_out={c_out.shape}")

    # Joint: encoder_output float32 [1,1024,1], predictor_output float32 [1,640,1] -> joint_output [1,1,1,8198]
    joint_sess = ort.InferenceSession(joint_path, sess_options=sess_opt, providers=providers)
    joint_inputs = {i.name: i for i in joint_sess.get_inputs()}
    if not {"encoder_output", "predictor_output"}.issubset(set(joint_inputs.keys())):
        raise RuntimeError(f"Joint inputs unexpected: {list(joint_inputs.keys())}")
    enc = np.zeros((1, 1024, 1), dtype=np.float32)
    pred = np.zeros((1, 640, 1), dtype=np.float32)
    (joint_out,) = joint_sess.run(None, {"encoder_output": enc, "predictor_output": pred})
    logger.info(f"ORT joint ok | joint_output={joint_out.shape} {joint_out.dtype}")

def export_tokenizer_assets(model, out_dir: str):
    """
    Best-effort export of tokenizer and vocab assets for offline TRT builds.
    """
    vocab = None
    if hasattr(model, "decoder") and hasattr(model.decoder, "vocabulary"):
        vocab = list(model.decoder.vocabulary)
    if vocab:
        vocab_path = os.path.join(out_dir, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for t in vocab:
                f.write(t)
                f.write("\n")
        logger.info(f"Wrote vocab: {vocab_path} ({len(vocab)} tokens)")

    # Try to dump SentencePiece model bytes if available.
    tok = getattr(model, "tokenizer", None)
    if tok is None and hasattr(model, "decoder"):
        tok = getattr(model.decoder, "tokenizer", None)

    if tok is None:
        logger.info("No tokenizer object found on model; skipping tokenizer.model export")
        return

    spm_path = os.path.join(out_dir, "tokenizer.model")
    try:
        # Common NeMo SentencePieceTokenizer shape: tokenizer.tokenizer is a SentencePieceProcessor.
        sp = getattr(tok, "tokenizer", None)
        if sp is not None and hasattr(sp, "serialized_model_proto"):
            data = sp.serialized_model_proto()
            with open(spm_path, "wb") as f:
                f.write(data)
            logger.info(f"Wrote SentencePiece model: {spm_path} ({len(data)} bytes)")
            return
    except Exception as e:
        logger.warning(f"Tokenizer SentencePiece dump failed: {e}")

    # Fallback: try to copy a known model path from config.
    cfg = getattr(model, "cfg", None)
    model_path = None
    try:
        if cfg is not None and hasattr(cfg, "tokenizer"):
            model_path = getattr(cfg.tokenizer, "model", None) or getattr(cfg.tokenizer, "model_path", None)
    except Exception:
        model_path = None
    if model_path and os.path.exists(model_path):
        import shutil
        shutil.copyfile(model_path, spm_path)
        logger.info(f"Copied SentencePiece model from cfg: {spm_path}")
    else:
        logger.info("Tokenizer present but could not locate SentencePiece bytes/path; skipping tokenizer.model export")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .nemo file")
    parser.add_argument("--out", type=str, default="out", help="Output directory")
    parser.add_argument("--component", type=str, choices=['encoder', 'predictor', 'joint', 'encoder_streaming', 'all'], default='all')
    parser.add_argument("--fixed", action='store_true', help="Export with fixed shapes (no dynamic axes)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Export device (default: cpu)")
    parser.add_argument("--smoke-test-ort", action="store_true", help="Run minimal ONNX Runtime smoke test after export")
    parser.add_argument("--streaming-cache-size", type=int, default=256, help="Streaming encoder cache size (last_channel_cache_size)")
    parser.add_argument("--streaming-chunk-size", type=str, default="", help="Override encoder.streaming_cfg.chunk_size (comma-separated)")
    parser.add_argument("--streaming-shift-size", type=str, default="", help="Override encoder.streaming_cfg.shift_size (comma-separated)")
    parser.add_argument("--streaming-cache-drop-size", type=int, default=-1, help="Override encoder.streaming_cfg.cache_drop_size (>=0)")
    parser.add_argument("--streaming-dummy-len", type=int, default=0, help="Override dummy chunk length for streaming export")
    parser.add_argument("--streaming-no-postprocess", action="store_true", help="Skip streaming_post_process in streaming encoder export")
    parser.add_argument("--streaming-skip-setup", action="store_true", help="Skip encoder.setup_streaming_params in streaming export")
    args = parser.parse_args()

    # Start Heartbeat
    heartbeat = ExportHeartbeat(interval=5)
    heartbeat.start()

    os.makedirs(args.out, exist_ok=True)
    
    heartbeat.set_stage("Loading model")
    typecheck.set_typecheck_enabled(False)
    model = nemo_asr.models.ASRModel.restore_from(args.model)
    model.eval()
    torch.set_grad_enabled(False)
    
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Export device: {device} (forcing CPU is recommended for reliability)")
    model = model.to(device)
    model.eval()
    logger.info("Model set to eval() and grad disabled; exports will run under torch.inference_mode()")

    # Infer key vocab/joint metadata (best-effort).
    tokenizer_vocab_size = None
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "vocabulary"):
            tokenizer_vocab_size = len(model.decoder.vocabulary)
    except Exception:
        tokenizer_vocab_size = None

    joint_vocab_size = None
    try:
        # RNNTJoint uses `num_classes_with_blank` as the output dim of the final linear.
        joint_vocab_size = int(getattr(model.joint, "num_classes_with_blank", 0)) or None
    except Exception:
        joint_vocab_size = None

    duration_values = None
    try:
        # NeMo TDT loss often exposes `durations` via loss internals; keep best-effort.
        loss = getattr(model.joint, "loss", None)
        inner = getattr(loss, "_loss", None)
        dv = getattr(inner, "durations", None)
        if dv is not None:
            duration_values = list(dv)
    except Exception:
        duration_values = None

    # Save metadata
    metadata = {
        "model_name": "parakeet-tdt-0.6b-v3",
        "sample_rate": model.cfg.preprocessor.get('sample_rate', 16000),
        "labels": model.decoder.vocabulary if hasattr(model.decoder, 'vocabulary') else [],
        "blank_id": int(getattr(getattr(model, "decoder", None), "blank_idx", 0)),
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "joint_vocab_size": joint_vocab_size,
        "duration_values": duration_values,
        "torch_version": torch.__version__,
        "tensor_layout_contract": {
            "encoder_input": "audio_signal: [B, n_mels, T]",
            "encoder_output": "encoder_output: [B, D_enc(=1024), T_enc]",
            "predictor_input": "y: [B, U], h/c: [L, B, H]",
            "predictor_output": "g: [B, H(=640), U] (transposed from NeMo [B,U,H])",
            "joint_input": "encoder_output: [B, 1024, T], predictor_output: [B, 640, U]",
            "joint_output": "joint_output: [B, T, U, V_joint(=8198)]",
        },
        "features": {
            "type": "log-mel",
            "n_fft": model.cfg.preprocessor.get('n_fft', 512),
            "n_mels": model.cfg.preprocessor.get('features', 128),
            "hop_length": int(model.cfg.preprocessor.get('window_stride', 0.01) * model.cfg.preprocessor.get('sample_rate', 16000)),
        }
    }
    with open(os.path.join(args.out, "model_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    export_tokenizer_assets(model, args.out)

    components = [args.component] if args.component != 'all' else ['encoder', 'predictor', 'joint']

    exported = []
    failed = False
    for comp in components:
        heartbeat.set_stage(f"Exporting {comp}")
        try:
            with torch.inference_mode():
                if comp == 'encoder':
                    export_encoder(model, args.out, device, dynamic=not args.fixed)
                    exported.append(os.path.join(args.out, "encoder.onnx"))
                elif comp == 'encoder_streaming':
                    chunk_size = _parse_int_list(args.streaming_chunk_size) if args.streaming_chunk_size else None
                    shift_size = _parse_int_list(args.streaming_shift_size) if args.streaming_shift_size else None
                    cache_drop_size = args.streaming_cache_drop_size if args.streaming_cache_drop_size >= 0 else None
                    export_encoder_streaming(
                        model,
                        args.out,
                        device,
                        dynamic=not args.fixed,
                        cache_size=args.streaming_cache_size,
                        cache_drop_size=cache_drop_size,
                        chunk_size=chunk_size,
                        shift_size=shift_size,
                        dummy_len=args.streaming_dummy_len,
                        use_post_process=not args.streaming_no_postprocess,
                        skip_setup=args.streaming_skip_setup,
                    )
                    exported.append(os.path.join(args.out, "encoder_streaming.onnx"))
                elif comp == 'predictor':
                    export_predictor(model, args.out, device)
                    exported.append(os.path.join(args.out, "predictor.onnx"))
                elif comp == 'joint':
                    export_joint(model, args.out, device)
                    exported.append(os.path.join(args.out, "joint.onnx"))
        except Exception as e:
            failed = True
            logger.error(f"FATAL ERROR exporting {comp}: {e}")
            import traceback
            traceback.print_exc()

    heartbeat.set_stage("Finished")
    heartbeat.stop()
    logger.info("=== Export artifact manifest ===")
    for p in exported:
        try:
            ins, outs = _onnx_io_summary(p)
            data_path = p + ".data"
            extra = f" (+external data: {data_path})" if os.path.exists(data_path) else ""
            logger.info(f"- {p}{extra}")
            logger.info(f"  inputs: {ins}")
            logger.info(f"  outputs: {outs}")
        except Exception as e:
            logger.warning(f"- {p}: failed to summarize IO: {e}")

    if args.smoke_test_ort and (os.path.exists(os.path.join(args.out, 'predictor.onnx')) and os.path.exists(os.path.join(args.out, 'joint.onnx'))):
        _smoke_test_onnxruntime(
            os.path.join(args.out, "predictor.onnx"),
            os.path.join(args.out, "joint.onnx"),
        )

    if failed:
        logger.error("Export failed (one or more components). Exiting with code 1.")
        sys.exit(1)
    logger.info("Export completed successfully.")

if __name__ == "__main__":
    main()
