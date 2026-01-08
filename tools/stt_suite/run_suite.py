#!/usr/bin/env python3
"""
run_suite.py - Deterministic STT Test Suite with ALSA Loopback

Runs a matrix of test variants (base, nopunct, nocache, nocache_nopunct)
against LibriSpeech WAVs using an isolated ALSA loopback "virtual mic".

IMPORTANT: This suite is designed to NOT disturb system audio.
All audio routing uses the dedicated ALSA loopback device.

Usage:
    # Setup loopback first (one-time per boot):
    sudo modprobe snd-aloop index=10 id=LoopSTT

    # Run suite:
    python run_suite.py \
        --manifest manifest.tsv \
        --cli-path ~/git/trt-asr-engine/rust/target/debug/cli \
        --model-dir ~/git/trt-asr-engine/models/parakeet \
        --rounds 2

Features:
    - Verifies each setup step before proceeding
    - Prints diagnostic information on failures
    - Collects taps, logs, and transcripts for each run
    - Supports multiple rounds for stability testing
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def color(text: str, c: str) -> str:
    """Wrap text in ANSI color codes."""
    return f"{c}{text}{Colors.RESET}"


def print_step(step: int, total: int, msg: str):
    """Print a step header."""
    print(f"\n{Colors.CYAN}[{step}/{total}]{Colors.RESET} {Colors.BOLD}{msg}{Colors.RESET}")


def print_ok(msg: str):
    print(f"  {color('OK', Colors.GREEN)}: {msg}")


def print_warn(msg: str):
    print(f"  {color('WARN', Colors.YELLOW)}: {msg}")


def print_error(msg: str):
    print(f"  {color('ERROR', Colors.RED)}: {msg}")


def print_debug(label: str, value: str):
    print(f"  {color(label, Colors.BLUE)}: {value}")


def run_cmd(cmd: List[str], timeout: int = 30, capture: bool = True) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -2, "", f"Command not found: {cmd[0]}"
    except Exception as e:
        return -3, "", str(e)


@dataclass
class LoopbackConfig:
    """ALSA loopback device configuration."""
    card_id: str = "LoopSTT"
    index: int = 10
    playback_device: str = "hw:LoopSTT,0,0"
    capture_device: str = "hw:LoopSTT,1,0"
    plug_playback: str = "plughw:LoopSTT,0,0"
    plug_capture: str = "plughw:LoopSTT,1,0"


@dataclass
class Variant:
    """Test variant configuration."""
    name: str
    env: Dict[str, str] = field(default_factory=dict)


# Define test variants
VARIANTS = [
    Variant("base", {}),
    Variant("nopunct", {"PARAKEET_DISABLE_PUNCT_SUPPRESSION": "1"}),
    Variant("nocache", {"PARAKEET_DISABLE_CACHE": "1"}),
    Variant("nocache_nopunct", {
        "PARAKEET_DISABLE_CACHE": "1",
        "PARAKEET_DISABLE_PUNCT_SUPPRESSION": "1"
    }),
]


def check_loopback_loaded() -> Tuple[bool, str]:
    """Check if ALSA loopback module is loaded with correct ID."""
    rc, out, err = run_cmd(["aplay", "-l"])
    if rc != 0:
        return False, f"aplay -l failed: {err}"

    if "LoopSTT" in out:
        return True, "LoopSTT card found"

    # Check if any loopback exists
    if "Loopback" in out:
        return False, "Generic Loopback found but not LoopSTT. Run: sudo modprobe snd-aloop index=10 id=LoopSTT"

    return False, "No loopback device found. Run: sudo modprobe snd-aloop index=10 id=LoopSTT"


def check_arecord_available() -> bool:
    """Check if arecord is available."""
    rc, _, _ = run_cmd(["which", "arecord"])
    return rc == 0


def check_aplay_available() -> bool:
    """Check if aplay is available."""
    rc, _, _ = run_cmd(["which", "aplay"])
    return rc == 0


def check_cli_available(cli_path: Path) -> Tuple[bool, str]:
    """Check if the CLI binary exists and is executable."""
    if not cli_path.exists():
        return False, f"CLI binary not found: {cli_path}"
    if not os.access(cli_path, os.X_OK):
        return False, f"CLI binary not executable: {cli_path}"
    return True, f"CLI found: {cli_path}"


def check_model_dir(model_dir: Path) -> Tuple[bool, str]:
    """Check if model directory exists and contains expected files."""
    if not model_dir.exists():
        return False, f"Model directory not found: {model_dir}"

    # Check for common model files
    expected = ["encoder.trt", "joint.trt", "predictor.trt"]
    found = []
    missing = []
    for f in expected:
        if (model_dir / f).exists():
            found.append(f)
        else:
            # Also check .plan extension
            if (model_dir / f.replace('.trt', '.plan')).exists():
                found.append(f)
            else:
                missing.append(f)

    if missing:
        # Try .engine extension
        missing2 = []
        for f in missing:
            if (model_dir / f.replace('.trt', '.engine')).exists():
                found.append(f)
            else:
                missing2.append(f)
        missing = missing2

    if missing:
        return False, f"Missing model files: {missing}"

    return True, f"Model directory OK ({len(found)} files found)"


def verify_wav(wav_path: Path) -> Tuple[bool, Dict]:
    """Verify WAV file properties using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,duration,codec_name",
        "-of", "json",
        str(wav_path)
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        return False, {"error": err}

    try:
        data = json.loads(out)
        streams = data.get("streams", [])
        if not streams:
            return False, {"error": "No audio streams found"}
        stream = streams[0]
        props = {
            "sample_rate": int(stream.get("sample_rate", 0)),
            "channels": int(stream.get("channels", 0)),
            "duration": float(stream.get("duration", 0)),
            "codec": stream.get("codec_name", "unknown")
        }
        ok = props["sample_rate"] == 16000 and props["channels"] == 1
        return ok, props
    except Exception as e:
        return False, {"error": str(e)}


def test_loopback_audio(config: LoopbackConfig, test_wav: Path, output_dir: Path) -> Tuple[bool, str]:
    """Test loopback by playing a WAV and capturing it.

    Returns (success, message).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    test_capture = output_dir / "loopback_test.wav"

    # Get WAV duration
    ok, props = verify_wav(test_wav)
    if not ok:
        return False, f"Test WAV invalid: {props}"

    duration = min(props["duration"], 3.0)  # Cap at 3 seconds

    # Start capture in background
    capture_cmd = [
        "arecord",
        "-D", config.plug_capture,
        "-r", "16000",
        "-c", "1",
        "-f", "S16_LE",
        "-d", str(int(duration) + 1),
        str(test_capture)
    ]

    try:
        capture_proc = subprocess.Popen(
            capture_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a bit for capture to start
        time.sleep(0.2)

        # Play the test WAV
        play_cmd = [
            "aplay",
            "-D", config.plug_playback,
            "-q",
            str(test_wav)
        ]
        rc, _, err = run_cmd(play_cmd, timeout=int(duration) + 5)
        if rc != 0:
            capture_proc.terminate()
            return False, f"aplay failed: {err}"

        # Wait for capture to complete
        capture_proc.wait(timeout=int(duration) + 3)

        # Verify captured audio
        if not test_capture.exists():
            return False, "Capture file not created"

        ok, props = verify_wav(test_capture)
        if not ok:
            return False, f"Captured audio invalid: {props}"

        if props["duration"] < duration * 0.8:
            return False, f"Captured audio too short: {props['duration']:.2f}s (expected ~{duration:.2f}s)"

        return True, f"Loopback test passed (captured {props['duration']:.2f}s)"

    except subprocess.TimeoutExpired:
        capture_proc.terminate()
        return False, "Capture process timed out"
    except Exception as e:
        return False, f"Loopback test failed: {e}"


def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load manifest TSV and return list of utterance dicts."""
    utterances = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                utterances.append({
                    'utt_id': parts[0],
                    'wav_path': parts[1],
                    'ref_text': parts[2]
                })
    return utterances


def run_cli_transcription(
    cli_path: Path,
    model_dir: Path,
    wav_path: Path,
    env: Dict[str, str],
    timeout: int = 120,
    verbose: bool = False,
    stream_sim: float = 0.5,
    no_sleep: bool = False,
    feature_norm: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Run CLI transcription and return (success, transcript, debug_output).

    Uses --stream-sim to chunk audio processing and avoid encoder profile limits.
    """
    cmd = [
        str(cli_path),
        str(wav_path),
        "--model-dir", str(model_dir),
        "--stream-sim", str(stream_sim),  # Chunk processing to avoid encoder max
        "--no-sleep" if no_sleep else "",
        "--verbose" if verbose else ""
    ]
    if feature_norm:
        cmd.extend(["--feature-norm", feature_norm])
    cmd = [c for c in cmd if c]  # Remove empty strings

    full_env = os.environ.copy()
    full_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env
        )

        # Parse transcript from output
        # Collect all Final: lines and concatenate (streaming mode outputs multiple)
        transcript_parts = []
        for line in result.stdout.split('\n'):
            if line.startswith("Transcript:"):
                text = line.replace("Transcript:", "").strip()
                if text:
                    transcript_parts.append(text)
            elif line.startswith("Final:"):
                text = line.replace("Final:", "").strip()
                if text and text != ".":  # Skip punctuation-only
                    transcript_parts.append(text)
            elif line.startswith("Partial:"):
                # Capture last partial if no final
                text = line.replace("Partial:", "").strip()
                if text and not transcript_parts:
                    transcript_parts = [text]

        transcript = " ".join(transcript_parts)

        debug_output = result.stderr if result.stderr else ""

        return result.returncode == 0, transcript, debug_output

    except subprocess.TimeoutExpired:
        return False, "", "CLI timed out"
    except Exception as e:
        return False, "", str(e)


def parse_debug_output(debug_text: str) -> Dict:
    """Parse debug output for key metrics."""
    metrics = {
        'nan_count': 0,
        'inf_count': 0,
        'blank_count': 0,
        'token_count': 0,
        'feature_frames': 0,
        'errors': []
    }

    for line in debug_text.split('\n'):
        if 'nan=' in line.lower():
            match = re.search(r'nan=(\d+)', line, re.IGNORECASE)
            if match:
                metrics['nan_count'] = int(match.group(1))
        if 'inf=' in line.lower():
            match = re.search(r'inf=(\d+)', line, re.IGNORECASE)
            if match:
                metrics['inf_count'] = int(match.group(1))
        if 'frames=' in line.lower() or 'frames:' in line.lower():
            match = re.search(r'frames[=:]?\s*(\d+)', line, re.IGNORECASE)
            if match:
                metrics['feature_frames'] = int(match.group(1))
        if 'error' in line.lower():
            metrics['errors'].append(line.strip())

    return metrics


def run_variant(
    variant: Variant,
    round_num: int,
    utterances: List[Dict],
    cli_path: Path,
    model_dir: Path,
    output_dir: Path,
    loopback: LoopbackConfig,
    use_loopback: bool = False,
    verbose: bool = False,
    stream_sim: float = 0.5,
    no_sleep: bool = False,
    feature_norm: Optional[str] = None,
) -> Dict:
    """Run a single variant for one round.

    Returns dict with results and metrics.
    """
    round_dir = output_dir / variant.name / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Setup tap directory
    tap_dir = round_dir / "taps"
    tap_dir.mkdir(exist_ok=True)

    # Build environment
    env = {
        "PARAKEET_MAX_FRAMES_PER_PUSH": "256",  # Always set for streaming stability
        "AUDIO_TAP_ENABLE": "1",
        "AUDIO_TAP_DIR": str(tap_dir),
        "AUDIO_TAP_FEATURES": "1",
    }
    lib_dir = Path(__file__).resolve().parents[2] / "cpp" / "build"
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else str(lib_dir)
    env.update(variant.env)

    results = {
        'variant': variant.name,
        'round': round_num,
        'utterances': [],
        'success_count': 0,
        'fail_count': 0,
        'empty_count': 0,
        'nan_total': 0,
        'total_time': 0.0
    }

    transcripts = []
    start_time = time.time()

    for i, utt in enumerate(utterances):
        utt_id = utt['utt_id']
        wav_path = Path(utt['wav_path'])

        if not wav_path.exists():
            print_error(f"WAV not found: {wav_path}")
            results['fail_count'] += 1
            continue

        # Run transcription
        success, transcript, debug_out = run_cli_transcription(
            cli_path,
            model_dir,
            wav_path,
            env,
            verbose=verbose,
            stream_sim=stream_sim,
            no_sleep=no_sleep,
            feature_norm=feature_norm,
        )

        metrics = parse_debug_output(debug_out)

        utt_result = {
            'utt_id': utt_id,
            'success': success,
            'transcript': transcript,
            'ref_text': utt['ref_text'],
            'metrics': metrics
        }

        if success and transcript:
            results['success_count'] += 1
            transcripts.append((utt_id, transcript))
        elif success and not transcript:
            results['empty_count'] += 1
            transcripts.append((utt_id, ""))
            if verbose:
                print_warn(f"{utt_id}: Empty transcript")
        else:
            results['fail_count'] += 1
            if verbose:
                print_error(f"{utt_id}: {debug_out[:100]}")

        results['nan_total'] += metrics['nan_count']
        results['utterances'].append(utt_result)

        # Save debug output
        debug_path = round_dir / f"{utt_id}_debug.log"
        with open(debug_path, 'w') as f:
            f.write(debug_out)

        # Progress
        if (i + 1) % 5 == 0 or (i + 1) == len(utterances):
            print(f"    Progress: {i + 1}/{len(utterances)} "
                  f"(ok={results['success_count']}, empty={results['empty_count']}, fail={results['fail_count']})")

    results['total_time'] = time.time() - start_time

    # Write transcripts
    transcripts_path = round_dir / "transcripts.tsv"
    with open(transcripts_path, 'w') as f:
        for utt_id, transcript in transcripts:
            f.write(f"{utt_id}\t{transcript}\n")

    # Write summary
    summary_path = round_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'variant': variant.name,
            'round': round_num,
            'env': env,
            'success_count': results['success_count'],
            'fail_count': results['fail_count'],
            'empty_count': results['empty_count'],
            'nan_total': results['nan_total'],
            'total_time': results['total_time'],
            'utterance_count': len(utterances)
        }, f, indent=2)

    return results


def print_suite_summary(all_results: List[Dict]):
    """Print summary table of all results."""
    print("\n" + "=" * 90)
    print(f"{Colors.BOLD}SUITE SUMMARY{Colors.RESET}")
    print("=" * 90)
    print(f"{'Variant':<20} {'Round':>6} {'OK':>6} {'Empty':>6} {'Fail':>6} {'NaNs':>8} {'Time':>10}")
    print("-" * 90)

    for r in all_results:
        status_color = Colors.GREEN if r['fail_count'] == 0 and r['empty_count'] == 0 else \
                       Colors.YELLOW if r['fail_count'] == 0 else Colors.RED
        print(f"{r['variant']:<20} {r['round']:>6} "
              f"{color(str(r['success_count']), Colors.GREEN):>15} "
              f"{color(str(r['empty_count']), Colors.YELLOW):>15} "
              f"{color(str(r['fail_count']), Colors.RED):>15} "
              f"{r['nan_total']:>8} "
              f"{r['total_time']:>9.1f}s")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic STT Test Suite with ALSA Loopback"
    )
    parser.add_argument('--manifest', '-m', type=Path, required=True,
                        help="Path to manifest.tsv")
    parser.add_argument('--cli-path', type=Path, required=True,
                        help="Path to CLI binary")
    parser.add_argument('--model-dir', type=Path, required=True,
                        help="Path to model directory")
    parser.add_argument('--output-dir', '-o', type=Path, default=None,
                        help="Output directory (default: /tmp/stt_suite/<timestamp>)")
    parser.add_argument('--rounds', '-r', type=int, default=1,
                        help="Number of rounds per variant (default: 1)")
    parser.add_argument('--variants', '-V', nargs='+', default=None,
                        help="Variants to run (default: all)")
    parser.add_argument('--num-utterances', '-n', type=int, default=None,
                        help="Limit to N utterances")
    parser.add_argument('--use-loopback', action='store_true',
                        help="Use ALSA loopback for playback (requires setup)")
    parser.add_argument('--skip-checks', action='store_true',
                        help="Skip pre-flight checks")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Verbose output")
    parser.add_argument('--test-loopback', action='store_true',
                        help="Only test loopback setup, don't run suite")
    parser.add_argument('--stream-sim', type=float, default=0.5,
                        help="Streaming interval in seconds (default: 0.5)")
    parser.add_argument('--no-sleep', action='store_true',
                        help="Disable sleeping between streaming chunks")
    parser.add_argument('--feature-norm', choices=['none', 'per_feature'], default=None,
                        help="Feature normalization mode passed to CLI")

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("STT TEST SUITE - Deterministic Testing with Debug Diagnostics")
    print(f"{'=' * 60}{Colors.RESET}\n")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/tmp/stt_suite/suite_{timestamp}_{os.getpid()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    loopback = LoopbackConfig()
    total_steps = 6 if args.use_loopback else 5

    if not args.skip_checks:
        # Step 1: Check loopback (if needed)
        if args.use_loopback or args.test_loopback:
            print_step(1, total_steps, "Checking ALSA Loopback")
            ok, msg = check_loopback_loaded()
            if ok:
                print_ok(msg)
            else:
                print_error(msg)
                print_debug("FIX", "sudo modprobe snd-aloop index=10 id=LoopSTT")
                if args.test_loopback:
                    sys.exit(1)
                print_warn("Continuing without loopback...")
                args.use_loopback = False

            if ok and check_aplay_available() and check_arecord_available():
                print_ok("aplay and arecord available")
            elif ok:
                print_error("aplay or arecord not found")
                print_debug("FIX", "sudo apt install alsa-utils")
                if args.test_loopback:
                    sys.exit(1)

        # Step 2: Check CLI
        step = 2 if args.use_loopback else 1
        print_step(step, total_steps, "Checking CLI Binary")
        ok, msg = check_cli_available(args.cli_path)
        if ok:
            print_ok(msg)
        else:
            print_error(msg)
            print_debug("FIX", f"cargo build -p cli (in trt-asr-engine)")
            sys.exit(1)

        # Step 3: Check model directory
        step += 1
        print_step(step, total_steps, "Checking Model Directory")
        ok, msg = check_model_dir(args.model_dir)
        if ok:
            print_ok(msg)
        else:
            print_error(msg)
            sys.exit(1)

        # Step 4: Check manifest
        step += 1
        print_step(step, total_steps, "Loading Manifest")
        if not args.manifest.exists():
            print_error(f"Manifest not found: {args.manifest}")
            print_debug("FIX", "python make_manifest.py /path/to/LibriSpeech/dev-clean")
            sys.exit(1)

        utterances = load_manifest(args.manifest)
        if not utterances:
            print_error("Manifest is empty or invalid")
            sys.exit(1)

        print_ok(f"Loaded {len(utterances)} utterances")

        if args.num_utterances:
            utterances = utterances[:args.num_utterances]
            print_ok(f"Limited to {len(utterances)} utterances")

        # Verify first WAV
        first_wav = Path(utterances[0]['wav_path'])
        if not first_wav.exists():
            print_error(f"First WAV not found: {first_wav}")
            sys.exit(1)

        ok, props = verify_wav(first_wav)
        if ok:
            print_ok(f"First WAV valid: {props['sample_rate']}Hz, {props['channels']}ch, {props['duration']:.2f}s")
        else:
            print_error(f"First WAV invalid: {props}")
            sys.exit(1)

        # Step 5: Test loopback (if used)
        if args.use_loopback:
            step += 1
            print_step(step, total_steps, "Testing Loopback Audio Path")
            ok, msg = test_loopback_audio(loopback, first_wav, output_dir / "loopback_test")
            if ok:
                print_ok(msg)
            else:
                print_error(msg)
                if args.test_loopback:
                    sys.exit(1)
                print_warn("Continuing without loopback...")
                args.use_loopback = False

        if args.test_loopback:
            print(f"\n{Colors.GREEN}Loopback test complete!{Colors.RESET}")
            sys.exit(0)

    else:
        utterances = load_manifest(args.manifest)
        if args.num_utterances:
            utterances = utterances[:args.num_utterances]

    # Step 6: Run suite
    print_step(total_steps, total_steps, "Running Test Suite")

    # Filter variants
    variants_to_run = VARIANTS
    if args.variants:
        variants_to_run = [v for v in VARIANTS if v.name in args.variants]
        if not variants_to_run:
            print_error(f"No matching variants: {args.variants}")
            print_debug("Available", ", ".join(v.name for v in VARIANTS))
            sys.exit(1)

    print_ok(f"Variants: {[v.name for v in variants_to_run]}")
    print_ok(f"Rounds: {args.rounds}")
    print_ok(f"Utterances: {len(utterances)}")

    all_results = []

    for variant in variants_to_run:
        for round_num in range(args.rounds):
            print(f"\n  {Colors.MAGENTA}Running: {variant.name} (round {round_num}){Colors.RESET}")
            print(f"    Env: {variant.env}")

            results = run_variant(
                variant=variant,
                round_num=round_num,
                utterances=utterances,
                cli_path=args.cli_path,
                model_dir=args.model_dir,
                output_dir=output_dir,
                loopback=loopback,
                use_loopback=args.use_loopback,
                verbose=args.verbose,
                stream_sim=args.stream_sim,
                no_sleep=args.no_sleep,
                feature_norm=args.feature_norm,
            )
            all_results.append(results)

            # Print quick summary
            status = "OK" if results['fail_count'] == 0 else "FAIL"
            status_color = Colors.GREEN if results['fail_count'] == 0 else Colors.RED
            print(f"    {color(status, status_color)}: "
                  f"{results['success_count']} ok, "
                  f"{results['empty_count']} empty, "
                  f"{results['fail_count']} fail, "
                  f"{results['nan_total']} NaNs, "
                  f"{results['total_time']:.1f}s")

    # Print final summary
    print_suite_summary(all_results)

    # Write overall results
    results_path = output_dir / "all_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Check for failures
    total_fails = sum(r['fail_count'] for r in all_results)
    total_nans = sum(r['nan_total'] for r in all_results)
    total_empty = sum(r['empty_count'] for r in all_results)

    print(f"\n{Colors.BOLD}DIAGNOSTICS:{Colors.RESET}")
    if total_fails > 0:
        print(f"  {color('FAILURES', Colors.RED)}: {total_fails} utterances failed")
        print(f"    Check debug logs in: {output_dir}/<variant>/round_N/*_debug.log")
    if total_nans > 0:
        print(f"  {color('NaNs', Colors.YELLOW)}: {total_nans} NaN occurrences detected")
        print(f"    This may indicate numerical instability in encoder/joint")
    if total_empty > 0:
        print(f"  {color('EMPTY', Colors.YELLOW)}: {total_empty} empty transcripts")
        print(f"    Check if punct suppression is causing this (try nopunct variant)")

    # Exit code
    if total_fails > 0:
        sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}Suite completed successfully!{Colors.RESET}")
        sys.exit(0)


if __name__ == '__main__':
    main()
