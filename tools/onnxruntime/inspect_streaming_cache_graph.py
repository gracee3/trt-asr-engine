#!/usr/bin/env python3
import argparse
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import onnx
from onnx import numpy_helper


def load_onnx(path: str):
    return onnx.load(path)


def build_maps(model) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, onnx.TensorProto], Set[str]]:
    prod = {}
    for node in model.graph.node:
        for out in node.output:
            prod[out] = node
    inits = {init.name: init for init in model.graph.initializer}
    graph_inputs = {i.name for i in model.graph.input}
    return prod, inits, graph_inputs


def const_value(node: onnx.NodeProto) -> Optional[str]:
    for attr in node.attribute:
        if attr.name == "value":
            try:
                arr = numpy_helper.to_array(attr.t)
                if arr.size == 1:
                    return str(arr.reshape(-1)[0])
                preview = arr.reshape(-1)[:8]
                return f"{preview.tolist()} (shape={list(arr.shape)})"
            except Exception:
                return None
    return None


def initializer_value(init: onnx.TensorProto) -> Optional[str]:
    try:
        arr = numpy_helper.to_array(init)
        if arr.size == 1:
            return str(arr.reshape(-1)[0])
        preview = arr.reshape(-1)[:8]
        return f"{preview.tolist()} (shape={list(arr.shape)})"
    except Exception:
        return None


def trace_output(name: str,
                 prod: Dict[str, onnx.NodeProto],
                 inits: Dict[str, onnx.TensorProto],
                 graph_inputs: Set[str],
                 max_depth: int) -> None:
    print(f"\n=== Trace for {name} (depth={max_depth}) ===")
    seen = set()
    q = deque([(name, 0)])
    while q:
        cur, depth = q.popleft()
        key = (cur, depth)
        if key in seen:
            continue
        seen.add(key)
        if cur in graph_inputs:
            print(f"{'  '*depth}input: {cur}")
            continue
        if cur in inits:
            val = initializer_value(inits[cur])
            print(f"{'  '*depth}initializer: {cur} value={val}")
            continue
        node = prod.get(cur)
        if node is None:
            print(f"{'  '*depth}unresolved: {cur}")
            continue
        const = None
        if node.op_type == "Constant":
            const = const_value(node)
        header = f"{'  '*depth}{node.op_type} name={node.name or '<anon>'} -> {list(node.output)}"
        if const is not None:
            header += f" value={const}"
        print(header)
        if depth >= max_depth:
            continue
        for inp in node.input:
            if not inp:
                continue
            q.append((inp, depth + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect encoder_streaming cache outputs provenance.")
    ap.add_argument("--onnx", default="tools/export_onnx/out/encoder_streaming.onnx")
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()

    model = load_onnx(args.onnx)
    prod, inits, graph_inputs = build_maps(model)

    targets = [
        "cache_last_channel_len_out",
        "cache_last_channel_out",
        "cache_last_time_out",
    ]
    for t in targets:
        trace_output(t, prod, inits, graph_inputs, args.depth)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
