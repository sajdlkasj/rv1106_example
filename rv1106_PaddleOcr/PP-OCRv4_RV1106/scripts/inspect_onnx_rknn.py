#!/usr/bin/env python3
"""Inspect ONNX structure and RKNN conversion logs without modifying models.

Examples:
  python inspect_onnx_rknn.py --model models/onnx/det/static.onnx \
      --log logs/convert_det704.log
  python inspect_onnx_rknn.py --model models/onnx/ppocrv4_rec.onnx \
      --log logs/convert_rec_i8.log --show-nodes LayerNorm ReduceMean Pow Sqrt
"""

import argparse
import collections
import pathlib
import re
import sys

import onnx


def dtype_name(value):
    try:
        return onnx.TensorProto.DataType.Name(value)
    except ValueError:
        return str(value)


def tensor_shape(value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None, True, dtype_name(tensor_type.elem_type)
    dims = []
    dynamic = False
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(str(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            dims.append(dim.dim_param)
            dynamic = True
        else:
            dims.append("?")
            dynamic = True
    return dims, dynamic, dtype_name(tensor_type.elem_type)


def inspect_model(path, requested_ops):
    model = onnx.load(str(path))
    print(f"\n===== ONNX: {path} =====")
    try:
        onnx.checker.check_model(model)
        print("checker: PASS")
    except Exception as exc:
        print(f"checker: FAIL: {type(exc).__name__}: {exc}")

    print(f"IR version: {model.ir_version}")
    print("opset:", ", ".join(
        f"{entry.domain or 'ai.onnx'}={entry.version}" for entry in model.opset_import
    ))

    has_dynamic = False
    for label, values in (("input", model.graph.input), ("output", model.graph.output)):
        for value in values:
            dims, dynamic, dtype = tensor_shape(value)
            has_dynamic |= dynamic
            shape_text = "<missing>" if dims is None else f"[{', '.join(dims)}]"
            print(f"{label}: {value.name}  dtype={dtype}  shape={shape_text}  dynamic={dynamic}")
    print(f"graph has dynamic/unknown public dimension: {has_dynamic}")

    counts = collections.Counter(node.op_type for node in model.graph.node)
    print(f"raw node count: {len(model.graph.node)}; operator types: {len(counts)}")
    for op_type, count in sorted(counts.items()):
        print(f"  {op_type}: {count}")

    if requested_ops:
        requested = set(requested_ops)
        print("selected node details:")
        found = 0
        for index, node in enumerate(model.graph.node):
            if node.op_type not in requested:
                continue
            found += 1
            print(
                f"  #{index} name={node.name or '<unnamed>'} op={node.op_type} "
                f"inputs={list(node.input)} outputs={list(node.output)}"
            )
        if not found:
            print("  none")


TABLE_RE = re.compile(r"\]\s+(\d+)\s+(\S+)\s+\S+\s+(CPU|NPU)\s+")


def inspect_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    nodes = []
    warnings = []
    errors = []
    success_markers = []

    for line_no, line in enumerate(text.splitlines(), 1):
        match = TABLE_RE.search(line)
        if match:
            nodes.append({
                "id": int(match.group(1)),
                "op": match.group(2),
                "target": match.group(3),
                "line": line_no,
            })
        if "W RKNN:" in line or re.search(r"\bWARNING\b", line, re.I):
            warnings.append((line_no, line.strip()))
        if "E RKNN:" in line or "Traceback" in line or re.search(r"\b(build|load_onnx|export_rknn) failed\b", line, re.I):
            errors.append((line_no, line.strip()))
        if re.search(r"SUCCESS:|Conversion complete|RKNN models are ready|Export Mini RKNN model", line):
            success_markers.append((line_no, line.strip()))

    print(f"\n===== RKNN log: {path} =====")
    if nodes:
        target_counts = collections.Counter(node["target"] for node in nodes)
        print("device table:", dict(target_counts))
        cpu_nodes = [node for node in nodes if node["target"] == "CPU"]
        print("CPU-assigned nodes:")
        for node in cpu_nodes:
            print(f"  id={node['id']} op={node['op']} log_line={node['line']}")
        internal_cpu = [node for node in cpu_nodes if node["op"] not in {"InputOperator", "OutputOperator"}]
        print(f"internal CPU fallback count: {len(internal_cpu)}")
        op_counts = collections.Counter(node["op"] for node in nodes if node["target"] == "NPU")
        print("NPU operator summary:")
        for op, count in sorted(op_counts.items()):
            print(f"  {op}: {count}")
    else:
        print("device table: not found in this log; do not infer full-NPU placement from build success alone")

    print(f"warnings: {len(warnings)}")
    for line_no, line in warnings:
        print(f"  L{line_no}: {line}")
    print(f"error-looking lines: {len(errors)}")
    for line_no, line in errors:
        print(f"  L{line_no}: {line}")
    print(f"success/export markers: {len(success_markers)}")
    for line_no, line in success_markers[-10:]:
        print(f"  L{line_no}: {line}")

    if errors and success_markers:
        print("interpretation: the log contains error-level text and later export/success evidence; inspect device placement and board execution before deciding whether conversion failed")
    elif errors:
        print("interpretation: no later success/export marker was recognized; verify return codes and output file before using the model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], help="ONNX model path; repeatable")
    parser.add_argument("--log", action="append", default=[], help="RKNN conversion log path; repeatable")
    parser.add_argument("--show-nodes", nargs="*", default=[], help="print ONNX nodes whose op_type is listed")
    args = parser.parse_args()

    if not args.model and not args.log:
        parser.error("provide at least one --model or --log")

    failed = False
    for raw in args.model:
        path = pathlib.Path(raw)
        if not path.is_file():
            print(f"missing model: {path}", file=sys.stderr)
            failed = True
            continue
        inspect_model(path, args.show_nodes)
    for raw in args.log:
        path = pathlib.Path(raw)
        if not path.is_file():
            print(f"missing log: {path}", file=sys.stderr)
            failed = True
            continue
        inspect_log(path)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
