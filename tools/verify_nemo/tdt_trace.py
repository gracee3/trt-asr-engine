#!/usr/bin/env python3
import argparse
import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

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


def load_vocab_ids(vocab_path: str, tokens: List[str]) -> Dict[str, int]:
    mapping = {t: -1 for t in tokens}
    if not vocab_path or not os.path.exists(vocab_path):
        return mapping
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tok = line.rstrip("\r\n")
            if tok in mapping and mapping[tok] < 0:
                mapping[tok] = idx
    return mapping


def load_features_f32(path: str, n_mels: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % n_mels != 0:
        raise RuntimeError(f"features size {raw.size} not divisible by n_mels={n_mels}")
    t = raw.size // n_mels
    return raw.reshape(n_mels, t)


def infer_pred_state(model) -> Tuple[int, int]:
    num_layers, hidden_size = 2, 640
    try:
        pred = getattr(model.decoder, "prediction", None)
        if pred is not None and hasattr(pred, "__getitem__"):
            dec_rnn = pred["dec_rnn"]
            lstm = getattr(dec_rnn, "lstm", None)
            if lstm is not None:
                num_layers = int(getattr(lstm, "num_layers", num_layers) or num_layers)
                hidden_size = int(getattr(lstm, "hidden_size", hidden_size) or hidden_size)
    except Exception:
        pass
    return num_layers, hidden_size


@contextmanager
def force_joint_logits(joint: torch.nn.Module):
    orig = None
    try:
        if hasattr(joint, "log_softmax"):
            orig = getattr(joint, "log_softmax", None)
            setattr(joint, "log_softmax", False)
        yield
    finally:
        if orig is not None and hasattr(joint, "log_softmax"):
            setattr(joint, "log_softmax", orig)


@contextmanager
def disable_joint_fuse_loss_wer(joint: torch.nn.Module):
    orig = None
    try:
        orig = getattr(joint, "_fuse_loss_wer", None)
        if hasattr(joint, "set_fuse_loss_wer"):
            joint.set_fuse_loss_wer(False)
        elif orig is not None:
            setattr(joint, "_fuse_loss_wer", False)
        yield
    finally:
        if orig is not None:
            try:
                if hasattr(joint, "set_fuse_loss_wer"):
                    joint.set_fuse_loss_wer(bool(orig))
                else:
                    setattr(joint, "_fuse_loss_wer", orig)
            except Exception:
                pass


def predictor_step(model, y: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(model.decoder, "predict"):
        g, state = model.decoder.predict(y=y, state=[h, c], add_sos=False)
        g = g.transpose(1, 2).contiguous()
        return g, state[0].contiguous(), state[1].contiguous()
    target_length = torch.full((y.shape[0],), y.shape[1], dtype=torch.long, device=y.device)
    g, _, state = model.decoder(targets=y, target_length=target_length, states=[h, c])
    g = g.transpose(1, 2).contiguous()
    return g, state[0].contiguous(), state[1].contiguous()


def topk_list(vec: np.ndarray, k: int) -> List[Dict[str, float]]:
    if k <= 0:
        return []
    idx = np.argpartition(vec, -k)[-k:]
    idx = idx[np.argsort(vec[idx])[::-1]]
    out = []
    for i in idx:
        out.append({"local": int(i), "v": float(vec[i])})
    return out


def load_contract_heads(path: str) -> Tuple[int, int, List[int], int]:
    if not path or not os.path.exists(path):
        return 0, 8193, [0, 1, 2, 3, 4], 8192
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = data["joint"]["io"]["outputs"][0]
    tok = out["token_head"]
    dur = out["duration_head"]
    duration_values = data["joint"]["architecture"]["duration_values"]
    blank_id = data["joint"]["architecture"]["blank_id"]
    return int(tok["offset"]), int(tok["size"]), list(duration_values), int(blank_id)


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit a per-step TDT decode trace from PyTorch (logits-level).")
    ap.add_argument("--model", default="models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo")
    ap.add_argument("--model-dir", default="models/parakeet-tdt-0.6b-v3", help="Used to locate vocab.txt if --vocab not set")
    ap.add_argument("--vocab", default="", help="Path to vocab.txt (optional)")
    ap.add_argument("--features-f32", required=True, help="Path to f32le features [C,T]")
    ap.add_argument("--n-mels", type=int, default=0)
    ap.add_argument("--chunk-frames", type=int, default=0)
    ap.add_argument("--stream-sim", type=float, default=0.0, help="Chunk size in seconds (if chunk-frames not set)")
    ap.add_argument("--hop-ms", type=float, default=10.0)
    ap.add_argument("--drop-last", action="store_true", help="Drop remainder frames instead of running a short final chunk")
    ap.add_argument("--contract", default="contracts/parakeet-tdt-0.6b-v3.contract.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = all")
    ap.add_argument("--max-symbols", type=int, default=8)
    ap.add_argument("--token-topk", type=int, default=5)
    ap.add_argument("--dur-topk", type=int, default=5)
    ap.add_argument("--y0-override", type=int, default=-1)
    ap.add_argument("--no-prime", action="store_true", help="Skip NeMo-style start/lang priming")
    ap.add_argument("--out", required=True, help="Output JSONL trace path")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    model = load_model(args.model)
    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    n_mels = args.n_mels or int(model.cfg.preprocessor.get("features", 128))
    feats = load_features_f32(args.features_f32, n_mels)

    if args.chunk_frames > 0:
        chunk_frames = args.chunk_frames
    else:
        if args.stream_sim <= 0:
            raise RuntimeError("Provide --chunk-frames or --stream-sim")
        chunk_frames = int(round((args.stream_sim * 1000.0) / args.hop_ms))
        if chunk_frames <= 0:
            raise RuntimeError("Computed chunk_frames <= 0")

    vocab_path = args.vocab
    if not vocab_path:
        cand = os.path.join(args.model_dir, "vocab.txt")
        if os.path.exists(cand):
            vocab_path = cand
    priming = load_vocab_ids(vocab_path, ["<|startoftranscript|>", "<|en|>"])
    tok_start = priming.get("<|startoftranscript|>", -1)
    tok_lang = priming.get("<|en|>", -1)

    tok_offset, tok_size, duration_values, blank_id = load_contract_heads(args.contract)
    dur_offset = tok_offset + tok_size
    dur_size = len(duration_values)

    num_layers, hidden_size = infer_pred_state(model)
    h = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32, device=device)
    c = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32, device=device)

    def prime_token(tok: int, h_in: torch.Tensor, c_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = torch.tensor([[tok]], dtype=torch.long, device=device)
        g, h_out, c_out = predictor_step(model, y, h_in, c_in)
        return g, h_out, c_out

    y_id = blank_id
    g = None
    if args.y0_override >= 0:
        g, h, c = prime_token(args.y0_override, h, c)
        y_id = args.y0_override
    elif not args.no_prime:
        if tok_start >= 0:
            g, h, c = prime_token(tok_start, h, c)
            y_id = tok_start
        if tok_lang >= 0:
            g, h, c = prime_token(tok_lang, h, c)
            y_id = tok_lang
    if g is None:
        g, h, c = prime_token(blank_id, h, c)
        y_id = blank_id

    emitted: List[int] = []
    step_idx = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        header = {
            "type": "meta",
            "token_offset": tok_offset,
            "token_size": tok_size,
            "duration_offset": dur_offset,
            "duration_size": dur_size,
            "duration_values": duration_values,
            "blank_id": blank_id,
            "chunk_frames": chunk_frames,
            "y0": y_id,
        }
        out_f.write(json.dumps(header) + "\n")

        with torch.inference_mode(), disable_joint_fuse_loss_wer(model.joint), force_joint_logits(model.joint):
            total_frames = feats.shape[1]
            num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_frames
                end = min(start + chunk_frames, total_frames)
                if end - start < chunk_frames and args.drop_last:
                    break
                chunk = feats[:, start:end]
                x = torch.from_numpy(chunk).unsqueeze(0).to(device)
                x_len = torch.tensor([chunk.shape[1]], dtype=torch.long, device=device)

                try:
                    enc_out, enc_len = model.encoder(x, x_len)
                except Exception:
                    enc_out, enc_len = model.encoder(audio_signal=x, length=x_len)
                t_enc = int(enc_len.item())

                time_idx = 0
                while time_idx < t_enc:
                    advanced_time = False
                    for u in range(args.max_symbols):
                        enc_slice = enc_out[:, :, time_idx : time_idx + 1]
                        joint = model.joint(encoder_outputs=enc_slice, decoder_outputs=g)
                        logits = joint[0, 0, 0].detach().cpu().numpy()

                        tok_logits = logits[tok_offset : tok_offset + tok_size]
                        dur_logits = logits[dur_offset : dur_offset + dur_size]
                        best_tok = int(np.argmax(tok_logits))
                        best_tok_v = float(tok_logits[best_tok])
                        best_dur_idx = int(np.argmax(dur_logits))
                        duration = int(duration_values[best_dur_idx])
                        advance = duration
                        blank_dur0_clamped = False
                        if best_tok == blank_id and duration == 0:
                            advance = 1
                            blank_dur0_clamped = True

                        record = {
                            "type": "step",
                            "step_idx": step_idx,
                            "chunk_idx": chunk_idx,
                            "time_idx": time_idx,
                            "u": u,
                            "y_id": int(y_id),
                            "best_tok": best_tok,
                            "best_tok_v": best_tok_v,
                            "is_blank": bool(best_tok == blank_id),
                            "best_dur_idx": best_dur_idx,
                            "duration": duration,
                            "advance": advance,
                            "blank_dur0_clamped": bool(blank_dur0_clamped),
                            "tok_topk": topk_list(tok_logits, min(args.token_topk, tok_size)),
                            "dur_topk": topk_list(dur_logits, min(args.dur_topk, dur_size)),
                        }
                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        step_idx += 1

                        if best_tok != blank_id:
                            emitted.append(best_tok)
                            y = torch.tensor([[best_tok]], dtype=torch.long, device=device)
                            g, h, c = predictor_step(model, y, h, c)
                            y_id = best_tok

                        if advance == 0:
                            continue

                        time_idx += advance
                        advanced_time = True
                        break

                    if not advanced_time:
                        time_idx += 1

                    if args.max_steps and step_idx >= args.max_steps:
                        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
