#!/usr/bin/env python3
"""Parse PPOCR_BENCH key/value lines and emit reproducible latency statistics."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


LINE_RE = re.compile(r"^(BENCH_[A-Z]+)\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")


def percentile(values: list[float], p: float) -> float:
    """NumPy-compatible linear percentile without an external dependency."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "std_population": round(statistics.pstdev(values), 3),
        "min": round(min(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p90": round(percentile(values, 0.90), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def parse(path: Path) -> dict[str, list[dict[str, float]]]:
    records: dict[str, list[dict[str, float]]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.match(raw.strip())
        if not match:
            continue
        tag, body = match.groups()
        item = {key: float(value) for key, value in KV_RE.findall(body)}
        records.setdefault(tag, []).append(item)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    records = parse(args.log)
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for tag, items in sorted(records.items()):
        keys = sorted({key for item in items for key in item})
        summary[tag] = {
            key: describe([item[key] for item in items if key in item]) for key in keys
        }

    output = {
        "source_log": str(args.log),
        "record_counts": {tag: len(items) for tag, items in sorted(records.items())},
        "statistics": summary,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
