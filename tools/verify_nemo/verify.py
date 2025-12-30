import argparse
import os
import torch
import nemo.collections.asr as nemo_asr

def main():
    parser = argparse.ArgumentParser(description="Verify NeMo model transcription locally.")
    parser.add_argument("--model", type=str, required=True, help="Path to .nemo file")
    parser.add_argument("--wav", type=str, help="Path to test .wav file (16k mono)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return

    print(f"Loading model from {args.model}...")
    # restore_from automatically detects the correct EncDec class
    try:
        model = nemo_asr.models.ASRModel.restore_from(args.model)
    except Exception as e:
        print(f"Error loading model with ASRModel: {e}")
        print("Falling back to EncDecRNNTBPEModel...")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model {model.__class__.__name__} loaded successfully on {device}.")

    if args.wav:
        if not os.path.exists(args.wav):
            print(f"Error: WAV file {args.wav} not found.")
            return

        print(f"Transcribing {args.wav}...")
        with torch.no_grad():
            # Standard transcribe call
            # Parakeet-TDT often returns a list of strings
            result = model.transcribe([args.wav])
            
            # Handle potential tuple return (from hybrid models)
            if isinstance(result, tuple):
                transcript = result[0][0]
            else:
                transcript = result[0]

            print("\n--- Final Transcript ---")
            print(transcript)
            print("------------------------\n")
    else:
        print("Provide a --wav file to perform transcription.")

if __name__ == "__main__":
    main()
