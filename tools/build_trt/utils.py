from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class BuildError(RuntimeError):
    pass


def read_json(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def which_or_raise(binary: str) -> str:
    p = shutil.which(binary)
    if not p:
        raise BuildError(
            f"Required binary not found in PATH: {binary}. "
            f"Install TensorRT and ensure `{binary}` is available."
        )
    return p


@dataclass(frozen=True)
class OnnxArtifact:
    name: str  # encoder|predictor|joint
    onnx_path: Path
    external_data_files: Tuple[str, ...]  # filenames as referenced by ONNX external_data 'location'


def _try_import_onnx():
    try:
        import onnx  # type: ignore

        return onnx
    except Exception as e:
        raise BuildError(
            "Python package `onnx` is required to inspect ONNX inputs/external-data references. "
            "Install it (e.g. `pip install onnx`)."
        ) from e


def detect_external_data_files(onnx_path: Path) -> Tuple[str, ...]:
    """
    Return referenced external data filenames (e.g. ('encoder.onnx.data',)) if any.
    """
    onnx = _try_import_onnx()
    m = onnx.load_model(str(onnx_path), load_external_data=False)

    # ONNX stores external tensor refs on initializers: data_location==EXTERNAL and an external_data list.
    files: List[str] = []
    for init in m.graph.initializer:
        try:
            if int(getattr(init, "data_location", 0)) != int(onnx.TensorProto.EXTERNAL):
                continue
            for kv in getattr(init, "external_data", []):
                if getattr(kv, "key", "") == "location" and getattr(kv, "value", ""):
                    files.append(str(kv.value))
        except Exception:
            continue

    # Unique, stable order
    uniq: List[str] = []
    for f in files:
        if f not in uniq:
            uniq.append(f)
    return tuple(uniq)


def stage_onnx_artifact(
    *,
    artifact: OnnxArtifact,
    staging_root: Path,
) -> Path:
    """
    Stage an ONNX + its external data sidecars (if referenced) into `staging_root/`.
    Returns staged ONNX path.
    """
    staging_root.mkdir(parents=True, exist_ok=True)
    staged_onnx = staging_root / artifact.onnx_path.name
    shutil.copyfile(artifact.onnx_path, staged_onnx)

    # External weights: must be in same directory as .onnx to be discovered by parser.
    for fname in artifact.external_data_files:
        src = artifact.onnx_path.parent / fname
        if not src.exists():
            raise BuildError(
                f"{artifact.name}.onnx references external weights `{fname}` but it was not found next to "
                f"the ONNX: {src}. If you used the exporter, ensure the `.onnx.data` sidecar was preserved."
            )
        dst = staging_root / fname
        shutil.copyfile(src, dst)

    # Back-compat: if exporter used the typical `<name>.onnx.data` but the ONNX doesn't explicitly list it,
    # also stage it when present to reduce footguns.
    default_sidecar = artifact.onnx_path.with_suffix(artifact.onnx_path.suffix + ".data")
    if default_sidecar.exists():
        dst = staging_root / default_sidecar.name
        if not dst.exists():
            shutil.copyfile(default_sidecar, dst)

    return staged_onnx


def onnx_input_rank_and_static_dims(onnx_path: Path) -> Dict[str, Tuple[int, List[Optional[int]]]]:
    """
    Returns input_name -> (rank, dims) where dims entries are int if statically known else None.
    Uses load_external_data=False (fast even for large external weight models).
    """
    onnx = _try_import_onnx()
    m = onnx.load_model(str(onnx_path), load_external_data=False)
    out: Dict[str, Tuple[int, List[Optional[int]]]] = {}

    for i in m.graph.input:
        t = i.type.tensor_type
        dims: List[Optional[int]] = []
        for d in t.shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        out[i.name] = (len(dims), dims)

    return out


def fmt_shape(dims: Iterable[int]) -> str:
    return "x".join(str(int(d)) for d in dims)


def run_cmd_capture(
    *,
    cmd: List[str],
    cwd: Optional[Path],
    log_path: Path,
    verbose: bool,
) -> Tuple[int, str]:
    """
    Run command, write combined stdout/stderr to log_path, return (exit_code, text).
    If verbose is True, stream output to console while also capturing.
    """
    ensure_dir(log_path.parent)
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    lines: List[str] = []
    with log_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            lines.append(line)
            f.write(line)
            if verbose:
                print(line, end="")

    rc = proc.wait()
    elapsed = time.time() - start
    text = "".join(lines)
    if rc != 0:
        tail = "".join(lines[-80:])
        raise BuildError(
            f"Command failed (exit={rc}, {elapsed:.1f}s): {' '.join(cmd)}\n"
            f"Log: {log_path}\n"
            f"--- last output ---\n{tail}\n--- end last output ---"
        )
    return rc, text


_RE_TRT_VERSION = re.compile(r"TensorRT\s+version:\s*([^\s]+)", re.IGNORECASE)
_RE_GPU = re.compile(r"GPU\s+(\d+):\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_RE_CUDA = re.compile(r"CUDA\s+version:\s*([^\s]+)", re.IGNORECASE)


def parse_trtexec_env(text: str) -> dict:
    trt = None
    m = _RE_TRT_VERSION.search(text)
    if m:
        trt = m.group(1)

    gpu = None
    m2 = _RE_GPU.search(text)
    if m2:
        gpu = m2.group(2).strip()

    cuda = None
    m3 = _RE_CUDA.search(text)
    if m3:
        cuda = m3.group(1)

    return {"tensorrt_version": trt, "cuda_version": cuda, "gpu": gpu}


def sizeof_bytes(path: Path) -> int:
    return int(path.stat().st_size)


def human_bytes(n: int) -> str:
    # Simple IEC-ish formatting
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(x)} {u}"
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def make_temp_build_dir(prefix: str = "trt_build_") -> Path:
    p = Path(tempfile.mkdtemp(prefix=prefix))
    return p


