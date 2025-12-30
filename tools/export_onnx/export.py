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
        # For large models (>2GB), check_model should be called with the path
        onnx.checker.check_model(path)
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

class PredictorWrapper(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
    def forward(self, targets, target_length):
        return self.predictor(targets=targets, target_length=target_length)

class JointWrapper(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint
    def forward(self, encoder_output, predictor_output):
        return self.joint(encoder_outputs=encoder_output, decoder_outputs=predictor_output)

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

    logger.info(f"Calling torch.onnx.export for Encoder...")
    torch.onnx.export(
        model.encoder,
        (dummy_input, dummy_len),
        path,
        input_names=['audio_signal', 'length'],
        output_names=['encoder_output', 'encoded_lengths'],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        verbose=False
    )
    logger.info(f"Finished writing: {path}")
    validate_onnx(path)

def export_predictor(model, out_dir, device):
    stage = "Exporting Predictor"
    logger.info(f"Starting: {stage}")
    
    wrapper = PredictorWrapper(model.decoder).to(device)
    dummy_targets = torch.zeros(1, 1).long().to(device)
    dummy_target_len = torch.ones(1).long().to(device)
    
    path = os.path.join(out_dir, "predictor.onnx")
    
    logger.info("Tracing Predictor...")
    traced_model = torch.jit.trace(wrapper, (dummy_targets, dummy_target_len))
    
    logger.info("Calling torch.onnx.export for Predictor (traced)...")
    torch.onnx.export(
        traced_model,
        (dummy_targets, dummy_target_len),
        path,
        input_names=['targets', 'target_length'],
        output_names=['predictor_output', 'output_lengths'],
        dynamic_axes={
            'targets': {0: 'batch', 1: 'token_len'},
            'predictor_output': {0: 'batch', 2: 'token_len'}
        },
        opset_version=18
    )
    logger.info(f"Finished writing: {path}")
    validate_onnx(path)

def export_joint(model, out_dir, device):
    stage = "Exporting Joint"
    logger.info(f"Starting: {stage}")
    
    wrapper = JointWrapper(model.joint).to(device)
    h_enc = model.encoder._output_dim if hasattr(model.encoder, '_output_dim') else 1024
    h_pred = model.decoder._output_dim if hasattr(model.decoder, '_output_dim') else 1024
    
    dummy_enc = torch.randn(1, h_enc, 1).to(device)
    dummy_pred = torch.randn(1, h_pred, 1).to(device)
    
    path = os.path.join(out_dir, "joint.onnx")
    
    logger.info("Tracing Joint...")
    traced_model = torch.jit.trace(wrapper, (dummy_enc, dummy_pred))
    
    logger.info("Calling torch.onnx.export for Joint (traced)...")
    torch.onnx.export(
        traced_model,
        (dummy_enc, dummy_pred),
        path,
        input_names=['encoder_output', 'predictor_output'],
        output_names=['logits', 'durations'],
        dynamic_axes={
            'encoder_output': {0: 'batch', 2: 'time'},
            'predictor_output': {0: 'batch', 2: 'token_len'},
            'logits': {0: 'batch', 1: 'time', 2: 'token_len'},
            'durations': {0: 'batch', 1: 'time', 2: 'token_len'}
        },
        opset_version=18
    )
    logger.info(f"Finished writing: {path}")
    validate_onnx(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .nemo file")
    parser.add_argument("--out", type=str, default="out", help="Output directory")
    parser.add_argument("--component", type=str, choices=['encoder', 'predictor', 'joint', 'all'], default='all')
    parser.add_argument("--fixed", action='store_true', help="Export with fixed shapes (no dynamic axes)")
    args = parser.parse_args()

    # Start Heartbeat
    heartbeat = ExportHeartbeat(interval=5)
    heartbeat.start()

    os.makedirs(args.out, exist_ok=True)
    
    heartbeat.set_stage("Loading model")
    typecheck.set_typecheck_enabled(False)
    model = nemo_asr.models.ASRModel.restore_from(args.model)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Save metadata
    metadata = {
        "model_name": "parakeet-tdt-0.6b-v3",
        "sample_rate": model.cfg.preprocessor.get('sample_rate', 16000),
        "labels": model.decoder.vocabulary if hasattr(model.decoder, 'vocabulary') else [],
        "features": {
            "type": "log-mel",
            "n_fft": model.cfg.preprocessor.get('n_fft', 512),
            "n_mels": model.cfg.preprocessor.get('features', 128),
            "hop_length": int(model.cfg.preprocessor.get('window_stride', 0.01) * model.cfg.preprocessor.get('sample_rate', 16000)),
        }
    }
    with open(os.path.join(args.out, "model_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    components = [args.component] if args.component != 'all' else ['encoder', 'predictor', 'joint']

    for comp in components:
        heartbeat.set_stage(f"Exporting {comp}")
        try:
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
