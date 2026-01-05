#!/usr/bin/env python3
"""
score_wer.py - Compute Word Error Rate (WER) for STT results

Computes WER per utterance and aggregates per variant.
Outputs scores.tsv and prints a summary table.

Usage:
    python score_wer.py results_dir/ --manifest manifest.tsv

Expected directory structure:
    results_dir/
        base/
            round_0/
                transcripts.tsv  (utt_id<TAB>hyp_text)
            round_1/
                transcripts.tsv
        nopunct/
            ...
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def normalize_text(text: str) -> List[str]:
    """Normalize text for WER computation.

    - Uppercase
    - Remove punctuation
    - Split by whitespace
    """
    # Remove common punctuation
    text = text.upper()
    for punct in '.,!?;:\'"()-[]{}':
        text = text.replace(punct, '')
    # Collapse whitespace and split
    return text.split()


def edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Compute edit distance and error counts.

    Returns (substitutions, insertions, deletions, total_ref_words)
    """
    n = len(ref)
    m = len(hyp)

    # dp[i][j] = (cost, subs, ins, dels)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)

    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)  # All insertions

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)  # All deletions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                # Match
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                sub_cost, sub_s, sub_i, sub_d = dp[i - 1][j - 1]
                sub = (sub_cost + 1, sub_s + 1, sub_i, sub_d)

                # Insertion (extra word in hyp)
                ins_cost, ins_s, ins_i, ins_d = dp[i][j - 1]
                ins = (ins_cost + 1, ins_s, ins_i + 1, ins_d)

                # Deletion (missing word in hyp)
                del_cost, del_s, del_i, del_d = dp[i - 1][j]
                dele = (del_cost + 1, del_s, del_i, del_d + 1)

                # Pick minimum
                dp[i][j] = min([sub, ins, dele], key=lambda x: x[0])

    cost, subs, ins, dels = dp[n][m]
    return (subs, ins, dels, n)


def compute_wer(ref_text: str, hyp_text: str) -> Dict:
    """Compute WER metrics for a single utterance.

    Returns dict with:
        - wer: Word Error Rate (0.0 to 1.0+)
        - substitutions, insertions, deletions
        - ref_words, hyp_words (counts)
        - errors (total)
    """
    ref_words = normalize_text(ref_text)
    hyp_words = normalize_text(hyp_text)

    if len(ref_words) == 0:
        # Edge case: empty reference
        return {
            'wer': 1.0 if len(hyp_words) > 0 else 0.0,
            'substitutions': 0,
            'insertions': len(hyp_words),
            'deletions': 0,
            'ref_words': 0,
            'hyp_words': len(hyp_words),
            'errors': len(hyp_words)
        }

    subs, ins, dels, n = edit_distance(ref_words, hyp_words)
    errors = subs + ins + dels
    wer = errors / n if n > 0 else 0.0

    return {
        'wer': wer,
        'substitutions': subs,
        'insertions': ins,
        'deletions': dels,
        'ref_words': n,
        'hyp_words': len(hyp_words),
        'errors': errors
    }


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    """Load reference texts from manifest.

    Returns dict mapping utt_id -> ref_text
    """
    refs = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                utt_id, wav_path, ref_text = parts[0], parts[1], parts[2]
                refs[utt_id] = ref_text
    return refs


def load_transcripts(transcripts_path: Path) -> Dict[str, str]:
    """Load hypotheses from transcripts.tsv.

    Returns dict mapping utt_id -> hyp_text
    """
    hyps = {}
    with open(transcripts_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                utt_id, hyp_text = parts[0], parts[1]
                hyps[utt_id] = hyp_text
            elif len(parts) == 1:
                # Empty hypothesis
                hyps[parts[0]] = ''
    return hyps


def find_results(results_dir: Path) -> Dict[str, Dict[int, Path]]:
    """Find all transcript files organized by variant and round.

    Returns dict: variant -> {round_num -> transcripts_path}
    """
    results = defaultdict(dict)

    for variant_dir in results_dir.iterdir():
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name

        for round_dir in variant_dir.iterdir():
            if not round_dir.is_dir():
                continue
            if not round_dir.name.startswith('round_'):
                continue

            try:
                round_num = int(round_dir.name.replace('round_', ''))
            except ValueError:
                continue

            transcripts_path = round_dir / 'transcripts.tsv'
            if transcripts_path.exists():
                results[variant][round_num] = transcripts_path

    return dict(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute WER scores for STT suite results"
    )
    parser.add_argument('results_dir', type=Path,
                        help="Directory containing variant/round subdirectories")
    parser.add_argument('--manifest', '-m', type=Path, required=True,
                        help="Path to manifest.tsv with reference texts")
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help="Output scores TSV (default: results_dir/scores.tsv)")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Print per-utterance scores")

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"[ERROR] Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (args.results_dir / 'scores.tsv')

    # Load references
    print(f"[INFO] Loading manifest: {args.manifest}")
    refs = load_manifest(args.manifest)
    print(f"[INFO] Loaded {len(refs)} reference utterances")

    # Find results
    results = find_results(args.results_dir)
    if not results:
        print(f"[ERROR] No results found in {args.results_dir}", file=sys.stderr)
        print("[DEBUG] Expected structure: results_dir/variant/round_N/transcripts.tsv")
        sys.exit(1)

    print(f"[INFO] Found variants: {list(results.keys())}")

    # Score all
    all_scores = []
    variant_stats = defaultdict(lambda: {
        'total_ref_words': 0,
        'total_errors': 0,
        'total_subs': 0,
        'total_ins': 0,
        'total_dels': 0,
        'count': 0,
        'wers': []
    })

    for variant in sorted(results.keys()):
        rounds = results[variant]
        for round_num in sorted(rounds.keys()):
            transcripts_path = rounds[round_num]
            hyps = load_transcripts(transcripts_path)

            for utt_id in sorted(hyps.keys()):
                if utt_id not in refs:
                    print(f"[WARN] No reference for {utt_id}", file=sys.stderr)
                    continue

                ref_text = refs[utt_id]
                hyp_text = hyps[utt_id]
                metrics = compute_wer(ref_text, hyp_text)

                all_scores.append({
                    'variant': variant,
                    'round': round_num,
                    'utt_id': utt_id,
                    'ref': ref_text,
                    'hyp': hyp_text,
                    **metrics
                })

                # Aggregate
                stats = variant_stats[variant]
                stats['total_ref_words'] += metrics['ref_words']
                stats['total_errors'] += metrics['errors']
                stats['total_subs'] += metrics['substitutions']
                stats['total_ins'] += metrics['insertions']
                stats['total_dels'] += metrics['deletions']
                stats['count'] += 1
                stats['wers'].append(metrics['wer'])

                if args.verbose:
                    print(f"  {variant}/round_{round_num}/{utt_id}: WER={metrics['wer']:.2%}")

    # Write scores TSV
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("variant\tround\tutt_id\twer\tsubs\tins\tdels\tref_words\thyp_words\tref\thyp\n")
        for s in all_scores:
            f.write(f"{s['variant']}\t{s['round']}\t{s['utt_id']}\t{s['wer']:.4f}\t")
            f.write(f"{s['substitutions']}\t{s['insertions']}\t{s['deletions']}\t")
            f.write(f"{s['ref_words']}\t{s['hyp_words']}\t{s['ref']}\t{s['hyp']}\n")

    print(f"\n[INFO] Wrote scores to: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("WER SUMMARY BY VARIANT")
    print("=" * 80)
    print(f"{'Variant':<20} {'Count':>8} {'WER':>10} {'Subs':>8} {'Ins':>8} {'Dels':>8} {'RefWords':>10}")
    print("-" * 80)

    for variant in sorted(variant_stats.keys()):
        stats = variant_stats[variant]
        if stats['total_ref_words'] > 0:
            overall_wer = stats['total_errors'] / stats['total_ref_words']
        else:
            overall_wer = 0.0

        print(f"{variant:<20} {stats['count']:>8} {overall_wer:>10.2%} "
              f"{stats['total_subs']:>8} {stats['total_ins']:>8} {stats['total_dels']:>8} "
              f"{stats['total_ref_words']:>10}")

    print("-" * 80)

    # Overall
    total_ref = sum(s['total_ref_words'] for s in variant_stats.values())
    total_err = sum(s['total_errors'] for s in variant_stats.values())
    total_count = sum(s['count'] for s in variant_stats.values())
    overall = total_err / total_ref if total_ref > 0 else 0.0
    print(f"{'OVERALL':<20} {total_count:>8} {overall:>10.2%}")
    print("=" * 80)

    # Per-variant stats
    print("\nPER-VARIANT DETAILED STATS:")
    for variant in sorted(variant_stats.keys()):
        stats = variant_stats[variant]
        wers = stats['wers']
        if wers:
            mean_wer = sum(wers) / len(wers)
            sorted_wers = sorted(wers)
            median_wer = sorted_wers[len(sorted_wers) // 2]
            min_wer = min(wers)
            max_wer = max(wers)
            print(f"  {variant}: mean={mean_wer:.2%} median={median_wer:.2%} "
                  f"min={min_wer:.2%} max={max_wer:.2%}")

    # Check for failures
    empty_hyps = [s for s in all_scores if s['hyp_words'] == 0]
    if empty_hyps:
        print(f"\n[WARN] {len(empty_hyps)} utterances with empty hypothesis:")
        for s in empty_hyps[:5]:
            print(f"  - {s['variant']}/round_{s['round']}/{s['utt_id']}")
        if len(empty_hyps) > 5:
            print(f"  ... and {len(empty_hyps) - 5} more")


if __name__ == '__main__':
    main()
