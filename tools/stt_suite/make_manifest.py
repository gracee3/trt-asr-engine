#!/usr/bin/env python3
"""
make_manifest.py - Build a deterministic manifest from LibriSpeech

Converts FLAC to 16kHz mono PCM16 WAV and creates manifest.tsv with:
    utt_id<TAB>wav_path<TAB>reference_text

Usage:
    python make_manifest.py /path/to/LibriSpeech/dev-clean \
        --output manifest.tsv \
        --wav-dir /tmp/librispeech_wav \
        --num-utterances 25
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def find_transcripts(librispeech_root: Path) -> List[Path]:
    """Find all *.trans.txt files recursively."""
    return sorted(librispeech_root.rglob("*.trans.txt"))


def parse_transcript_file(trans_path: Path) -> List[Tuple[str, str]]:
    """Parse a LibriSpeech transcript file.

    Returns list of (utt_id, text) tuples.
    """
    utterances = []
    with open(trans_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            utterances.append((utt_id, text))
    return utterances


def find_flac_for_utterance(trans_path: Path, utt_id: str) -> Path:
    """Find the FLAC file for a given utterance ID."""
    flac_path = trans_path.parent / f"{utt_id}.flac"
    return flac_path


def convert_flac_to_wav(flac_path: Path, wav_path: Path, verbose: bool = False) -> bool:
    """Convert FLAC to 16kHz mono PCM16 WAV using ffmpeg.

    Returns True on success, False on failure.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(flac_path),
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        str(wav_path)
    ]

    if verbose:
        print(f"  Converting: {flac_path.name} -> {wav_path.name}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] ffmpeg failed for {flac_path}: {result.stderr}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Please install ffmpeg.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Exception converting {flac_path}: {e}", file=sys.stderr)
        return False


def verify_wav(wav_path: Path) -> dict:
    """Verify WAV file properties using ffprobe.

    Returns dict with sample_rate, channels, duration, or None on failure.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels,duration',
        '-of', 'csv=p=0',
        str(wav_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(',')
        if len(parts) >= 2:
            return {
                'sample_rate': int(parts[0]),
                'channels': int(parts[1]),
                'duration': float(parts[2]) if len(parts) > 2 else None
            }
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build LibriSpeech manifest with FLAC->WAV conversion"
    )
    parser.add_argument('librispeech_root', type=Path,
                        help="Path to LibriSpeech subset (e.g., /path/to/dev-clean)")
    parser.add_argument('--output', '-o', type=Path, default=Path('manifest.tsv'),
                        help="Output manifest TSV path (default: manifest.tsv)")
    parser.add_argument('--wav-dir', type=Path, default=None,
                        help="Directory to store converted WAV files (default: next to manifest)")
    parser.add_argument('--num-utterances', '-n', type=int, default=None,
                        help="Limit to N utterances (default: all)")
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip conversion if WAV already exists")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Verbose output")
    parser.add_argument('--verify', action='store_true',
                        help="Verify each WAV after conversion")

    args = parser.parse_args()

    if not args.librispeech_root.exists():
        print(f"[ERROR] LibriSpeech root not found: {args.librispeech_root}", file=sys.stderr)
        sys.exit(1)

    # Default wav-dir next to manifest
    wav_dir = args.wav_dir or args.output.parent / 'wav'
    wav_dir = Path(wav_dir)

    print(f"[INFO] LibriSpeech root: {args.librispeech_root}")
    print(f"[INFO] Output manifest: {args.output}")
    print(f"[INFO] WAV directory: {wav_dir}")

    # Find all transcripts
    trans_files = find_transcripts(args.librispeech_root)
    if not trans_files:
        print(f"[ERROR] No transcript files found in {args.librispeech_root}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(trans_files)} transcript files")

    # Collect all utterances
    all_utterances = []
    for trans_path in trans_files:
        utterances = parse_transcript_file(trans_path)
        for utt_id, text in utterances:
            flac_path = find_flac_for_utterance(trans_path, utt_id)
            if flac_path.exists():
                all_utterances.append((utt_id, text, flac_path, trans_path))
            else:
                if args.verbose:
                    print(f"[WARN] FLAC not found: {flac_path}")

    print(f"[INFO] Found {len(all_utterances)} utterances with FLAC files")

    # Limit if requested
    if args.num_utterances:
        all_utterances = all_utterances[:args.num_utterances]
        print(f"[INFO] Limited to {len(all_utterances)} utterances")

    # Convert and build manifest
    manifest_rows = []
    converted = 0
    skipped = 0
    failed = 0
    verify_errors = []

    for utt_id, text, flac_path, trans_path in all_utterances:
        # Build WAV path preserving some directory structure
        rel_path = flac_path.relative_to(args.librispeech_root)
        wav_path = wav_dir / rel_path.with_suffix('.wav')

        if args.skip_existing and wav_path.exists():
            skipped += 1
            if args.verbose:
                print(f"  Skipping (exists): {utt_id}")
        else:
            success = convert_flac_to_wav(flac_path, wav_path, verbose=args.verbose)
            if not success:
                failed += 1
                continue
            converted += 1

        # Verify if requested
        if args.verify and wav_path.exists():
            props = verify_wav(wav_path)
            if props is None:
                verify_errors.append(f"{utt_id}: ffprobe failed")
            elif props['sample_rate'] != 16000:
                verify_errors.append(f"{utt_id}: sample_rate={props['sample_rate']} (expected 16000)")
            elif props['channels'] != 1:
                verify_errors.append(f"{utt_id}: channels={props['channels']} (expected 1)")

        manifest_rows.append((utt_id, str(wav_path.absolute()), text))

    # Write manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("utt_id\twav_path\tref_text\n")
        for utt_id, wav_path, text in manifest_rows:
            f.write(f"{utt_id}\t{wav_path}\t{text}\n")

    print(f"\n[SUMMARY]")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total rows: {len(manifest_rows)}")
    print(f"  Manifest: {args.output}")

    if verify_errors:
        print(f"\n[VERIFY ERRORS] {len(verify_errors)} issues:")
        for err in verify_errors[:10]:
            print(f"  - {err}")
        if len(verify_errors) > 10:
            print(f"  ... and {len(verify_errors) - 10} more")
        sys.exit(1)

    if failed > 0:
        print(f"\n[WARN] {failed} conversions failed")
        sys.exit(1)

    print("\n[OK] Manifest created successfully")


if __name__ == '__main__':
    main()
