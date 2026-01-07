#!/usr/bin/env python3
import argparse
import sys

import onnx


def _last_dim_value(output):
    if not output.type.HasField("tensor_type"):
        return None
    shape = output.type.tensor_type.shape
    if not shape.dim:
        return None
    dim = shape.dim[-1]
    if dim.HasField("dim_value"):
        return dim.dim_value
    if dim.HasField("dim_param"):
        return dim.dim_param
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check joint.onnx output layout and normalization ops.")
    parser.add_argument(
        "--model",
        default="tools/export_onnx/out/joint.onnx",
        help="Path to joint.onnx (default: tools/export_onnx/out/joint.onnx)",
    )
    parser.add_argument("--expect-vocab", type=int, default=8198, help="Expected joint vocab size")
    args = parser.parse_args()

    model = onnx.load(args.model)
    ops = {node.op_type for node in model.graph.node}
    has_logsoftmax = "LogSoftmax" in ops
    has_softmax = "Softmax" in ops

    output = model.graph.output[0] if model.graph.output else None
    out_name = output.name if output else "<missing>"
    last_dim = _last_dim_value(output) if output else None

    print(f"output_name={out_name}")
    print(f"output_last_dim={last_dim}")
    print(f"has_logsoftmax={has_logsoftmax}")
    print(f"has_softmax={has_softmax}")

    failures = []
    if has_logsoftmax or has_softmax:
        failures.append("joint graph contains softmax/logsoftmax")
    if isinstance(last_dim, int) and args.expect_vocab is not None and last_dim != args.expect_vocab:
        failures.append(f"expected vocab {args.expect_vocab}, got {last_dim}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
