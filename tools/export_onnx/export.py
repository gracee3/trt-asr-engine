import argparse
import os
import torch
import json
import nemo.collections.asr as nemo_asr
from nemo.core.classes import Exportable

def export(model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    # Load the model
    # Parakeet-TDT 0.6b-v3 is an EncDecRNNTBPEModel
    model = nemo_asr.models.ASRModel.restore_from(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model class: {model.__class__.__name__}")
    
    # Export Tokenizer/Vocab
    print("Exporting tokenizer/vocab...")
    if hasattr(model, 'tokenizer'):
        # For BPE models
        if hasattr(model.tokenizer, 'vocab'):
            with open(os.path.join(out_dir, "vocab.txt"), "w") as f:
                for token in model.tokenizer.vocab:
                    f.write(token + "\n")
        
        # If it's a SentencePiece tokenizer, try to get the .model file
        # NeMo often stores it in a temp dir or inside the .nemo tarball
        # We can try to extract it from the model config
        try:
            tokenizer_conf = model.cfg.tokenizer
            if 'model' in tokenizer_conf:
                # The model is likely extracted in the model's effective runtime dir
                # For now, let's just log and skip if not obvious
                print(f"Tokenizer model path in config: {tokenizer_conf.model}")
        except:
            pass

    # Export Metadata
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
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Attempting to export components...")
    from nemo.core.classes.common import typecheck
    typecheck.set_typecheck_enabled(False)
    # 1. Export Encoder
    print("Exporting Encoder...")
    encoder_path = os.path.join(out_dir, "encoder.onnx")
    try:
        n_mels = metadata["features"]["n_mels"]
        dummy_input = torch.randn(1, n_mels, 64).to(device)
        dummy_len = torch.LongTensor([64]).to(device)
        
        torch.onnx.export(
            model.encoder,
            (dummy_input, dummy_len),
            encoder_path,
            input_names=['audio_signal', 'length'],
            output_names=['encoder_output', 'encoded_lengths'],
            dynamic_axes={
                'audio_signal': {0: 'batch', 2: 'time'},
                'encoder_output': {0: 'batch', 2: 'time'}
            },
            opset_version=18
        )
        print(f"Encoder exported to {encoder_path}")
    except Exception as e:
        print(f"Encoder export failed: {e}")

    # 2. Export Predictor (Decoder)
    print("Exporting Predictor...")
    predictor_path = os.path.join(out_dir, "predictor.onnx")
    try:
        dummy_targets = torch.zeros(1, 1).long().to(device)
        dummy_target_len = torch.ones(1).long().to(device)
        
        torch.onnx.export(
            model.decoder,
            (dummy_targets, dummy_target_len),
            predictor_path,
            input_names=['targets', 'target_length'],
            output_names=['predictor_output', 'output_lengths'],
            dynamic_axes={
                'targets': {0: 'batch', 1: 'token_len'},
                'predictor_output': {0: 'batch', 2: 'token_len'}
            },
            opset_version=18
        )
        print(f"Predictor exported to {predictor_path}")
    except Exception as e:
        print(f"Predictor export failed: {e}")

    # 3. Export Joint
    print("Exporting Joint...")
    joint_path = os.path.join(out_dir, "joint.onnx")
    try:
        h_enc = model.encoder._output_dim if hasattr(model.encoder, '_output_dim') else 1024
        h_pred = model.decoder._output_dim if hasattr(model.decoder, '_output_dim') else 1024
        
        dummy_enc = torch.randn(1, h_enc, 1).to(device)
        dummy_pred = torch.randn(1, h_pred, 1).to(device)
        
        torch.onnx.export(
            model.joint,
            (dummy_enc, dummy_pred),
            joint_path,
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
        print(f"Joint exported to {joint_path}")
    except Exception as e:
        print(f"Joint export failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .nemo file")
    parser.add_argument("--out", type=str, default="out", help="Output directory")
    args = parser.parse_args()
    
    export(args.model, args.out)
