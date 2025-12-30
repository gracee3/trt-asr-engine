#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect/smoke-run a TensorRT engine via trtexec.")
    ap.add_argument("--engine", required=True, help="Path to *.engine")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shapes", default=None, help="Shape string for dynamic engines, e.g. name:1x2x3,name2:4x5")
    ap.add_argument("--trtexec", default="trtexec")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    engine = Path(args.engine).resolve()
    if not engine.exists():
        print(f"Engine not found: {engine}", file=sys.stderr)
        sys.exit(1)

    cmd = [args.trtexec, f"--loadEngine={engine}", f"--device={args.device}"]
    if args.shapes:
        cmd.append(f"--shapes={args.shapes}")
    cmd += ["--iterations=1", "--warmUp=0", "--duration=0"]
    if args.verbose:
        cmd.append("--verbose")

    p = subprocess.run(cmd, text=True)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()


