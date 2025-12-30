#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import profiles
from utils import (
    BuildError,
    OnnxArtifact,
    detect_external_data_files,
    ensure_dir,
    human_bytes,
    onnx_input_rank_and_static_dims,
    parse_trtexec_env,
    read_json,
    run_cmd_capture,
    sizeof_bytes,
    stage_onnx_artifact,
    which_or_raise,
)


def _trtexec_help(trtexec: str) -> str:
    import subprocess

    p = subprocess.run([trtexec, "--help"], capture_output=True, text=True)
    return (p.stdout or "") + "\n" + (p.stderr or "")


def _supports(help_text: str, needle: str) -> bool:
    return needle in help_text


def _resolve_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_onnx_paths(meta_path: Path, onnx_dir_override: Optional[str]) -> Dict[str, Path]:
    onnx_dir = Path(onnx_dir_override) if onnx_dir_override else meta_path.parent
    out = {}
    for name in ("encoder", "predictor", "joint"):
        p = onnx_dir / f"{name}.onnx"
        if not p.exists():
            raise BuildError(
                f"Missing ONNX artifact: {p}\n"
                f"Expected `{name}.onnx` next to the meta file ({meta_path.parent}) or in --onnx-dir."
            )
        out[name] = p
    return out


def _infer_predictor_l_h(predictor_onnx: Path) -> Tuple[int, int]:
    ins = onnx_input_rank_and_static_dims(predictor_onnx)
    if "h" not in ins:
        raise BuildError(f"predictor.onnx missing expected input `h`. Inputs: {list(ins.keys())}")
    _, dims = ins["h"]
    if len(dims) != 3:
        raise BuildError(f"predictor.onnx `h` rank expected 3, got {len(dims)} dims={dims}")

    L = dims[0]
    H = dims[2]
    if L is None or H is None:
        # Exporter uses 2 layers, 640 hidden by default. Keep this as a robust fallback.
        L = L or 2
        H = H or 640
    return int(L), int(H)


def _infer_encoder_length_rank(encoder_onnx: Path) -> int:
    ins = onnx_input_rank_and_static_dims(encoder_onnx)
    if "length" not in ins:
        raise BuildError(f"encoder.onnx missing expected input `length`. Inputs: {list(ins.keys())}")
    rank, _ = ins["length"]
    return int(rank)


def _fmt_shapes_for_smoke(opt_shapes: Dict[str, Tuple[int, ...]]) -> str:
    # trtexec expects the same "name:1x2,name2:3x4" format
    parts = [f"{k}:{'x'.join(str(int(d)) for d in v)}" for k, v in opt_shapes.items()]
    return ",".join(parts)


def build_one(
    *,
    name: str,
    trtexec: str,
    trtexec_help: str,
    staged_onnx: Path,
    outdir: Path,
    profile: profiles.ShapeProfile,
    device: int,
    fp16: bool,
    workspace_mb: int,
    optimization_level: int,
    timing_cache: Path,
    verbose: bool,
) -> dict:
    engine_path = outdir / f"{name}.engine"
    log_dir = ensure_dir(outdir / "build_logs")
    build_log = log_dir / f"{name}.log"

    cmd: List[str] = [trtexec]

    # Prefer explicit batch when supported (older/newer TRT differs; it's safe to omit if not present).
    if _supports(trtexec_help, "--explicitBatch"):
        cmd.append("--explicitBatch")

    cmd += [
        f"--device={device}",
        f"--onnx={staged_onnx}",
        f"--saveEngine={engine_path}",
        f"--builderOptimizationLevel={optimization_level}",
        f"--timingCacheFile={timing_cache}",
    ]

    # Workspace flag name varies by TRT version. Try the most common.
    if _supports(trtexec_help, "--workspace"):
        cmd.append(f"--workspace={workspace_mb}")
    elif _supports(trtexec_help, "--memPoolSize"):
        cmd.append(f"--memPoolSize=workspace:{workspace_mb}")
    else:
        # If neither exists, skip; build may still succeed with defaults.
        pass

    if fp16:
        cmd.append("--fp16")

    for k, v in profile.trtexec_flags().items():
        cmd.append(f"{k}={v}")

    t0 = time.time()
    _, out_text = run_cmd_capture(cmd=cmd, cwd=staged_onnx.parent, log_path=build_log, verbose=verbose)
    t_build = time.time() - t0

    if not engine_path.exists():
        raise BuildError(f"{name} build completed but engine was not written: {engine_path}")

    size = sizeof_bytes(engine_path)
    if size < 1024 * 1024:
        raise BuildError(f"{name} engine size looks too small ({human_bytes(size)}): {engine_path}")

    # Smoke run at opt shapes
    smoke_log = log_dir / f"{name}_smoke.log"
    smoke_cmd: List[str] = [trtexec, f"--loadEngine={engine_path}", f"--device={device}"]
    if _supports(trtexec_help, "--iterations"):
        smoke_cmd.append("--iterations=1")
    if _supports(trtexec_help, "--warmUp"):
        smoke_cmd.append("--warmUp=0")
    if _supports(trtexec_help, "--duration"):
        smoke_cmd.append("--duration=0")
    if _supports(trtexec_help, "--shapes"):
        smoke_cmd.append(f"--shapes={_fmt_shapes_for_smoke(profile.opt_shapes)}")
    else:
        # Shouldn't happen on modern trtexec; without shapes a dynamic engine can't run.
        raise BuildError("trtexec does not appear to support --shapes; cannot smoke-test dynamic engine.")

    t1 = time.time()
    _, smoke_out = run_cmd_capture(cmd=smoke_cmd, cwd=engine_path.parent, log_path=smoke_log, verbose=verbose)
    t_smoke = time.time() - t1

    env = parse_trtexec_env(out_text + "\n" + smoke_out)

    return {
        "component": name,
        "onnx": str(staged_onnx),
        "engine": str(engine_path),
        "commands": {
            "build": cmd,
            "smoke": smoke_cmd,
        },
        "engine_bytes": size,
        "engine_human": human_bytes(size),
        "profile": {
            "min": {k: list(v) for k, v in profile.min_shapes.items()},
            "opt": {k: list(v) for k, v in profile.opt_shapes.items()},
            "max": {k: list(v) for k, v in profile.max_shapes.items()},
        },
        "timings_sec": {"build": round(t_build, 3), "smoke": round(t_smoke, 3)},
        "logs": {"build": str(build_log), "smoke": str(smoke_log)},
        "env": env,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TensorRT engines from Parakeet ONNX components via trtexec.")
    ap.add_argument("--meta", required=True, help="Path to tools/export_onnx/out/model_meta.json")
    ap.add_argument("--onnx-dir", default=None, help="Override directory containing encoder/predictor/joint.onnx")
    ap.add_argument("--outdir", required=True, help="Output directory for *.engine and logs")
    ap.add_argument("--device", type=int, default=0, help="GPU device index (default: 0)")
    ap.add_argument("--workspace-mb", type=int, default=4096, help="Builder workspace in MB (default: 4096)")
    ap.add_argument("--fp16", action="store_true", help="Enable FP16")
    ap.add_argument("--builder-opt-level", type=int, default=5, help="TensorRT builderOptimizationLevel (default: 5)")
    ap.add_argument("--timing-cache", default=None, help="Timing cache path (default: <outdir>/timing.cache)")
    ap.add_argument("--trtexec", default="trtexec", help="Path to trtexec (default: trtexec in PATH)")
    ap.add_argument("--verbose", action="store_true", help="Stream full trtexec output to console")
    ap.add_argument("--keep-temp", action="store_true", help="Keep staged /tmp directory on success (and failure)")

    # Profile overrides (phase 1 defaults)
    ap.add_argument("--encoder-min-T", type=int, default=16)
    ap.add_argument("--encoder-opt-T", type=int, default=64)
    ap.add_argument("--encoder-max-T", type=int, default=256)

    ap.add_argument("--predictor-min-U", type=int, default=1)
    ap.add_argument("--predictor-opt-U", type=int, default=1)
    ap.add_argument("--predictor-max-U", type=int, default=8)

    ap.add_argument("--joint-min-T", type=int, default=16)
    ap.add_argument("--joint-opt-T", type=int, default=64)
    ap.add_argument("--joint-max-T", type=int, default=256)
    ap.add_argument("--joint-min-U", type=int, default=1)
    ap.add_argument("--joint-opt-U", type=int, default=1)
    ap.add_argument("--joint-max-U", type=int, default=8)

    args = ap.parse_args()

    meta_path = Path(args.meta).resolve()
    if not meta_path.exists():
        raise BuildError(f"Meta file not found: {meta_path}")

    meta = read_json(meta_path)
    n_mels = int(meta.get("features", {}).get("n_mels", 128) or 128)
    # Contract constants for parakeet-tdt-0.6b-v3
    d_enc = 1024
    d_pred = 640
    b = 1  # Phase 1: batch=1 only

    trtexec = which_or_raise(args.trtexec)
    help_text = _trtexec_help(trtexec)

    outdir = _resolve_outdir(args.outdir)
    timing_cache = Path(args.timing_cache).resolve() if args.timing_cache else (outdir / "timing.cache")

    # Resolve artifacts from meta dir unless overridden
    onnx_paths = _resolve_onnx_paths(meta_path, args.onnx_dir)

    artifacts: Dict[str, OnnxArtifact] = {}
    for name, p in onnx_paths.items():
        ext = detect_external_data_files(p)
        artifacts[name] = OnnxArtifact(name=name, onnx_path=p, external_data_files=ext)

    # Infer rank/dims needed for profile correctness
    length_rank = _infer_encoder_length_rank(onnx_paths["encoder"])
    pred_L, pred_H = _infer_predictor_l_h(onnx_paths["predictor"])

    encoder_prof = profiles.encoder_profile(
        b=b,
        n_mels=n_mels,
        t_min=args.encoder_min_T,
        t_opt=args.encoder_opt_T,
        t_max=args.encoder_max_T,
        length_rank=length_rank,
    )
    predictor_prof = profiles.predictor_profile(
        b=b,
        u_min=args.predictor_min_U,
        u_opt=args.predictor_opt_U,
        u_max=args.predictor_max_U,
        num_layers=pred_L,
        hidden_size=pred_H,
    )
    joint_prof = profiles.joint_profile(
        b=b,
        d_enc=d_enc,
        d_pred=d_pred,
        t_min=args.joint_min_T,
        t_opt=args.joint_opt_T,
        t_max=args.joint_max_T,
        u_min=args.joint_min_U,
        u_opt=args.joint_opt_U,
        u_max=args.joint_max_U,
    )

    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    staging_root = Path(f"/tmp/trt_build_{ts}").resolve()
    if staging_root.exists():
        # Extremely unlikely; fall back to unique suffix
        staging_root = Path(f"/tmp/trt_build_{ts}_{os.getpid()}").resolve()
    staging_root.mkdir(parents=True, exist_ok=False)

    report: dict = {
        "meta": str(meta_path),
        "onnx_dir": str((Path(args.onnx_dir) if args.onnx_dir else meta_path.parent).resolve()),
        "outdir": str(outdir),
        "timing_cache": str(timing_cache),
        "device": args.device,
        "fp16": bool(args.fp16),
        "builder_optimization_level": int(args.builder_opt_level),
        "workspace_mb": int(args.workspace_mb),
        "trtexec": {"path": trtexec},
        "staging_dir": str(staging_root),
        "started_utc": ts,
        "components": [],
    }

    try:
        # Stage artifacts into a clean directory for external-data discovery.
        staged = {}
        for name, art in artifacts.items():
            staged[name] = stage_onnx_artifact(artifact=art, staging_root=staging_root)

        # Build in dependency-ish order
        report["components"].append(
            build_one(
                name="encoder",
                trtexec=trtexec,
                trtexec_help=help_text,
                staged_onnx=staged["encoder"],
                outdir=outdir,
                profile=encoder_prof,
                device=args.device,
                fp16=args.fp16,
                workspace_mb=args.workspace_mb,
                optimization_level=args.builder_opt_level,
                timing_cache=timing_cache,
                verbose=args.verbose,
            )
        )
        report["components"].append(
            build_one(
                name="predictor",
                trtexec=trtexec,
                trtexec_help=help_text,
                staged_onnx=staged["predictor"],
                outdir=outdir,
                profile=predictor_prof,
                device=args.device,
                fp16=args.fp16,
                workspace_mb=args.workspace_mb,
                optimization_level=args.builder_opt_level,
                timing_cache=timing_cache,
                verbose=args.verbose,
            )
        )
        report["components"].append(
            build_one(
                name="joint",
                trtexec=trtexec,
                trtexec_help=help_text,
                staged_onnx=staged["joint"],
                outdir=outdir,
                profile=joint_prof,
                device=args.device,
                fp16=args.fp16,
                workspace_mb=args.workspace_mb,
                optimization_level=args.builder_opt_level,
                timing_cache=timing_cache,
                verbose=args.verbose,
            )
        )

        # Consolidate env (first non-null wins)
        env = {"tensorrt_version": None, "cuda_version": None, "gpu": None}
        for c in report["components"]:
            for k in env.keys():
                if env[k] is None and c.get("env", {}).get(k):
                    env[k] = c["env"][k]
        report["env"] = env
        report["finished_utc"] = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        report_path = outdir / "build_report.json"
        import json

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Concise summary to stdout
        print("=== TensorRT build report ===")
        print(f"outdir: {outdir}")
        if env.get("tensorrt_version"):
            print(f"TensorRT: {env['tensorrt_version']}")
        if env.get("cuda_version"):
            print(f"CUDA: {env['cuda_version']}")
        if env.get("gpu"):
            print(f"GPU: {env['gpu']}")
        for c in report["components"]:
            print(f"- {c['component']}: {c['engine_human']} | build={c['timings_sec']['build']}s smoke={c['timings_sec']['smoke']}s")
        print(f"report: {report_path}")

    except BuildError as e:
        # Persist report stub for CI visibility
        try:
            report["error"] = str(e)
            import json

            report_path = outdir / "build_report.json"
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass
        raise
    finally:
        if args.keep_temp:
            print(f"Kept staging dir: {staging_root}")
        else:
            # Best-effort cleanup
            try:
                import shutil

                shutil.rmtree(staging_root)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except BuildError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

