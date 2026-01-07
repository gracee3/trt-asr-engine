#!/usr/bin/env python3
"""
Build a pinned LibriSpeech gate manifest (dev-clean + dev-other).

Outputs a TSV with columns:
  utt_id \t wav_path \t ref_text \t split \t sha256

WAVs are generated under eval/wav/librispeech_dev_gate/<split>/...
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def find_transcripts(split_root: Path) -> List[Path]:
    return sorted(split_root.rglob("*.trans.txt"))


def parse_transcript_file(trans_path: Path) -> List[Tuple[str, str]]:
    utterances: List[Tuple[str, str]] = []
    with trans_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            utterances.append((parts[0], parts[1]))
    return utterances


def convert_flac_to_wav(flac_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(flac_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {flac_path}: {result.stderr}")
        return

    raise RuntimeError("ffmpeg not found; install ffmpeg to convert LibriSpeech FLAC -> WAV")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_split_items(split_root: Path) -> List[Tuple[str, str, Path]]:
    items: Dict[str, Tuple[str, str, Path]] = {}
    trans_files = find_transcripts(split_root)
    if not trans_files:
        raise FileNotFoundError(f"No transcript files under {split_root}")

    for trans_path in trans_files:
        for utt_id, text in parse_transcript_file(trans_path):
            flac_path = trans_path.parent / f"{utt_id}.flac"
            if not flac_path.exists():
                continue
            items[utt_id] = (utt_id, text, flac_path)

    return [items[k] for k in sorted(items.keys())]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LibriSpeech dev-clean/dev-other gate manifest.")
    parser.add_argument(
        "--librispeech-root",
        required=True,
        help="Path containing dev-clean/ and dev-other/",
    )
    parser.add_argument("--num-per-split", type=int, default=50)
    parser.add_argument("--manifest", default=None, help="Output TSV path")
    parser.add_argument("--out-dir", default=None, help="Output WAV dir root")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest) if args.manifest else repo_root / "eval" / "manifests" / "librispeech_dev_gate.tsv"
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "eval" / "wav" / "librispeech_dev_gate"

    root = Path(args.librispeech_root).expanduser().resolve()
    splits = ["dev-clean", "dev-other"]

    rows: List[Tuple[str, Path, str, str, str]] = []
    for split in splits:
        split_root = root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split: {split_root}")

        items = collect_split_items(split_root)
        if len(items) < args.num_per_split:
            raise RuntimeError(f"Split {split} has only {len(items)} utterances; need {args.num_per_split}")

        for utt_id, text, flac_path in items[: args.num_per_split]:
            gate_id = f"{split}_{utt_id}"
            wav_path = out_dir / split / f"{utt_id}.wav"
            if not wav_path.exists():
                convert_flac_to_wav(flac_path, wav_path)
            sha256 = sha256_file(wav_path)
            rows.append((gate_id, wav_path, text, split, sha256))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("utt_id\twav_path\tref_text\tsplit\tsha256\n")
        for gate_id, wav_path, text, split, sha256 in rows:
            try:
                rel = wav_path.relative_to(repo_root)
                wav_str = str(rel)
            except ValueError:
                wav_str = str(wav_path)
            f.write(f"{gate_id}\t{wav_str}\t{text}\t{split}\t{sha256}\n")

    print(f"wrote_manifest={manifest_path}")
    print(f"wav_dir={out_dir}")
    print(f"num_utts={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
