#!/usr/bin/env python3
"""
Regression test: ensure our ONNX staging logic copies *all* referenced external-data blobs.

This caught a real-world failure where the encoder graph referenced an external blob via a Constant node attribute:
  Constant_2337_attr__value

Run (from repo root):
  python tools/build_trt/test_stage_external_data.py --onnx tools/export_onnx/out/encoder.onnx
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from utils import OnnxArtifact, detect_external_data_files, stage_onnx_artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to an ONNX file to stage (e.g. encoder.onnx)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise SystemExit(f"ONNX not found: {onnx_path}")

    ext = detect_external_data_files(onnx_path)
    if not ext:
        print(f"[ok] {onnx_path.name}: no external data references found")
        return

    artifact = OnnxArtifact(name=onnx_path.stem, onnx_path=onnx_path, external_data_files=ext)

    with tempfile.TemporaryDirectory(prefix="trt_stage_test_") as td:
        staging_root = Path(td)
        staged_onnx = stage_onnx_artifact(artifact=artifact, staging_root=staging_root)

        missing: list[str] = []
        for fname in ext:
            p = staging_root / fname
            if not p.exists():
                missing.append(fname)

        if missing:
            raise SystemExit(
                f"[FAIL] missing staged external blobs for {onnx_path.name}:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + f"\nStaged onnx: {staged_onnx}"
            )

        print(f"[ok] staged {onnx_path.name}:")
        print(f"  external blobs: {len(ext)}")
        for fname in ext[:20]:
            print(f"  - {fname}")
        if len(ext) > 20:
            print(f"  ... (+{len(ext) - 20} more)")


if __name__ == "__main__":
    main()


