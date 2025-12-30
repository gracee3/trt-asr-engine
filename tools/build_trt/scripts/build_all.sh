#!/usr/bin/env bash
set -euo pipefail

# One-button wrapper for building TensorRT engines.
#
# Minimal:
#   tools/build_trt/scripts/build_all.sh \
#     --meta tools/export_onnx/out/model_meta.json \
#     --outdir models/parakeet-tdt-0.6b-v3 \
#     --fp16
#
# Optional export step:
#   tools/build_trt/scripts/build_all.sh --export --nemo models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo ...

EXPORT=0
NEMO=""

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --export)
      EXPORT=1
      shift
      ;;
    --nemo)
      NEMO="${2:-}"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$EXPORT" -eq 1 ]]; then
  if [[ -z "$NEMO" ]]; then
    echo "ERROR: --export requires --nemo <path-to-model.nemo>" >&2
    exit 1
  fi
  python tools/export_onnx/export.py --model "$NEMO" --out tools/export_onnx/out --component all
fi

python tools/build_trt/build_trt.py "${ARGS[@]}"


