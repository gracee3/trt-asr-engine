import argparse
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
import nemo.collections.asr as nemo_asr
from nemo.core.classes.common import typecheck

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    parser.add_argument("--component", type=str, choices=['encoder', 'predictor', 'joint', 'all'], default='all')
    parser.add_argument("--fixed", action='store_true', help="Export with fixed shapes (no dynamic axes)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Export device (default: cpu)")
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

    # Save metadata
    metadata = {
        "model_name": "parakeet-tdt-0.6b-v3",
        "sample_rate": model.cfg.preprocessor.get('sample_rate', 16000),
        "labels": model.decoder.vocabulary if hasattr(model.decoder, 'vocabulary') else [],
        "blank_id": int(getattr(getattr(model, "decoder", None), "blank_idx", 0)),
        "torch_version": torch.__version__,
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

    for comp in components:
        heartbeat.set_stage(f"Exporting {comp}")
        try:
            with torch.inference_mode():
                if comp == 'encoder':
                    export_encoder(model, args.out, device, dynamic=not args.fixed)
                elif comp == 'predictor':
                    export_predictor(model, args.out, device)
                elif comp == 'joint':
                    export_joint(model, args.out, device)
        except Exception as e:
            logger.error(f"FATAL ERROR exporting {comp}: {e}")
            import traceback
            traceback.print_exc()

    heartbeat.set_stage("Finished")
    heartbeat.stop()
    print("Export process completed.")

if __name__ == "__main__":
    main()
