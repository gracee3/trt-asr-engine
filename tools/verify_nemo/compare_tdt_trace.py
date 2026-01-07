#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List


def load_pt_trace(path: str) -> List[Dict[str, Any]]:
    steps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "step":
                steps.append(rec)
    return steps


def load_cpp_trace(path: str) -> List[Dict[str, Any]]:
    steps = []
    if not os.path.exists(path):
        raise RuntimeError(f"cpp trace not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "location" in rec and "data" in rec:
                if "parakeet_push_features:tdt_step" in rec.get("location", ""):
                    data = rec.get("data", {})
                    steps.append(data)
    return steps


def load_cpp_stderr(path: str) -> List[Dict[str, Any]]:
    steps = []
    rx = re.compile(
        r"tdt_step time_idx=(\d+) u=(\d+) best_tok=(\d+) best_dur_idx=(\d+) duration=(\d+) advance=(\d+) blank=(\d) blank_dur0_clamped=(\d)"
    )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = rx.search(line)
            if not m:
                continue
            steps.append(
                {
                    "time_idx": int(m.group(1)),
                    "u": int(m.group(2)),
                    "best_tok": int(m.group(3)),
                    "best_dur_idx": int(m.group(4)),
                    "duration": int(m.group(5)),
                    "advance": int(m.group(6)),
                    "is_blank": bool(int(m.group(7))),
                    "blank_dur0_clamped": bool(int(m.group(8))),
                }
            )
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare PyTorch vs C++ TDT trace logs.")
    ap.add_argument("--pt-trace", required=True, help="PyTorch JSONL trace from tdt_trace.py")
    ap.add_argument("--cpp-trace", default="/home/emmy/git/trt-asr-engine/.cursor/debug.log", help="C++ NDJSON trace (dbglog)")
    ap.add_argument("--cpp-stderr", default="", help="Optional stderr log file if NDJSON not available")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = all")
    ap.add_argument("--fields", default="best_tok,best_dur_idx,advance", help="Comma-separated fields to compare")
    ap.add_argument("--check-index", action="store_true", help="Also require time_idx and u to match")
    args = ap.parse_args()

    pt_steps = load_pt_trace(args.pt_trace)
    if args.cpp_stderr:
        cpp_steps = load_cpp_stderr(args.cpp_stderr)
    else:
        cpp_steps = load_cpp_trace(args.cpp_trace)

    if not pt_steps:
        print("No PyTorch steps found.", file=sys.stderr)
        return 2
    if not cpp_steps:
        print("No C++ steps found.", file=sys.stderr)
        return 2

    n = min(len(pt_steps), len(cpp_steps))
    if args.max_steps > 0:
        n = min(n, args.max_steps)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    for i in range(n):
        pt = pt_steps[i]
        cpp = cpp_steps[i]
        if args.check_index:
            if pt.get("time_idx") != cpp.get("time_idx") or pt.get("u") != cpp.get("u"):
                print(f"Mismatch at step {i}: index pt=(t={pt.get('time_idx')},u={pt.get('u')}) "
                      f"cpp=(t={cpp.get('time_idx')},u={cpp.get('u')})")
                return 1
        for field in fields:
            if pt.get(field) != cpp.get(field):
                print(f"Mismatch at step {i} field={field}: pt={pt.get(field)} cpp={cpp.get(field)}")
                if "tok_topk" in pt or "tok_topk" in cpp:
                    print(f"  pt tok_topk={pt.get('tok_topk')}")
                    print(f"  cpp tok_topk={cpp.get('tok_topk')}")
                if "dur_topk" in pt or "dur_topk" in cpp:
                    print(f"  pt dur_topk={pt.get('dur_topk')}")
                    print(f"  cpp dur_topk={cpp.get('dur_topk')}")
                return 1

    print(f"OK: matched {n} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
