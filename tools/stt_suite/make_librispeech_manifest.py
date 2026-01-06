#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class Item:
    utt_id: str
    flac_path: Path
    ref_text: str

def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None

def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDERR:\n{p.stderr}")

def find_split_root(librispeech_root: Path, split: str) -> Path:
    # Accept either:
    #  - <root>/LibriSpeech/<split>
    #  - <root>/<split>
    cand1 = librispeech_root / "LibriSpeech" / split
    cand2 = librispeech_root / split
    if cand1.is_dir():
        return cand1
    if cand2.is_dir():
        return cand2
    raise FileNotFoundError(f"Could not find split dir. Tried: {cand1} and {cand2}")

def load_transcripts(chapter_dir: Path) -> Dict[str, str]:
    # Transcript files are like: <speaker>-<chapter>.trans.txt
    trans_files = list(chapter_dir.glob("*.trans.txt"))
    m: Dict[str, str] = {}
    for tf in trans_files:
        with tf.open("r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                # format: <utt_id> <TEXT...>
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                utt, txt = parts[0].strip(), parts[1].strip()
                if utt:
                    m[utt] = txt
    return m

def pick_items(split_root: Path, num_utts: int) -> List[Item]:
    flacs = sorted(split_root.rglob("*.flac"))
    if not flacs:
        raise FileNotFoundError(f"No .flac files found under {split_root}")

    items: List[Item] = []
    for flac in flacs:
        utt_id = flac.stem
        chapter_dir = flac.parent
        trans = load_transcripts(chapter_dir)
        ref = trans.get(utt_id, "").strip()
        if not ref:
            continue
        items.append(Item(utt_id=utt_id, flac_path=flac, ref_text=ref))
        if len(items) >= num_utts:
            break

    if len(items) < num_utts:
        raise RuntimeError(f"Found only {len(items)} usable utterances with transcripts; requested {num_utts}")
    return items

def convert_to_wav_16k_mono(flac_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer ffmpeg, then sox, then flac decode + sox.
    if have_cmd("ffmpeg"):
        # -loglevel error: clean output
        run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(flac_path),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(wav_path),
        ])
        return

    if have_cmd("sox"):
        run([
            "sox",
            str(flac_path),
            "-r", "16000",
            "-c", "1",
            "-b", "16",
            str(wav_path),
        ])
        return

    if have_cmd("flac") and have_cmd("sox"):
        tmp = wav_path.with_suffix(".tmp.wav")
        run(["flac", "-d", "-f", "-o", str(tmp), str(flac_path)])
        run(["sox", str(tmp), "-r", "16000", "-c", "1", "-b", "16", str(wav_path)])
        try:
            tmp.unlink()
        except Exception:
            pass
        return

    raise RuntimeError("No available converter. Install ffmpeg or sox.")

def wav_duration_sec(wav_path: Path) -> float:
    import wave
    with wave.open(str(wav_path), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        return float(n) / float(sr) if sr else 0.0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech-root", required=True, help="Path containing LibriSpeech/<split>/...")
    ap.add_argument("--split", default="dev-clean", help="e.g. dev-clean, test-clean, train-clean-100")
    ap.add_argument("--num-utts", type=int, default=8)
    ap.add_argument("--out-dir", required=True, help="Output dir for converted WAVs and metadata")
    ap.add_argument("--manifest", required=True, help="TSV manifest output path")
    args = ap.parse_args()

    root = Path(args.librispeech_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()

    split_root = find_split_root(root, args.split)
    items = pick_items(split_root, args.num_utts)

    wav_dir = out_dir / "wavs16k_mono"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest:
    # utt_id \t wav_path \t duration_sec \t ref_text
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        f.write("utt_id\twav_path\tduration_sec\tref_text\n")
        for it in items:
            wav_path = wav_dir / f"{it.utt_id}.wav"
            convert_to_wav_16k_mono(it.flac_path, wav_path)
            dur = wav_duration_sec(wav_path)
            f.write(f"{it.utt_id}\t{wav_path}\t{dur:.6f}\t{it.ref_text}\n")

    print(f"wrote_manifest={manifest}")
    print(f"wav_dir={wav_dir}")
    print(f"num_utts={len(items)} split_root={split_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
