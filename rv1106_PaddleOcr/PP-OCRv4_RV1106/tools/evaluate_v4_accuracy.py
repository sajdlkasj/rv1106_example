#!/usr/bin/env python3
"""Evaluate the manually aligned two-image PP-OCRv4 RV1106 sample."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def normalize_strict(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def normalize_content(text: str) -> str:
    strict = normalize_strict(text)
    return "".join(char for char in strict if unicodedata.category(char)[0] not in {"P", "S"})


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_char != hyp_char),
            ))
        previous = current
    return previous[-1]


def evaluate_rows_with(rows: list[dict], field: str, normalizer) -> dict:
    gt_chars = 0
    errors = 0
    exact = 0
    emitted = 0
    for row in rows:
        gt = normalizer(row["gt"])
        pred = normalizer(row[field])
        gt_chars += len(gt)
        errors += edit_distance(gt, pred)
        exact += int(gt == pred)
        emitted += int(bool(pred))
    count = len(rows)
    cer = errors / gt_chars if gt_chars else 0.0
    return {
        "gt_lines": count,
        "emitted_lines": emitted,
        "exact_lines": exact,
        "exact_line_rate": round(exact / count, 6) if count else 0.0,
        "gt_characters": gt_chars,
        "edit_errors": errors,
        "cer": round(cer, 6),
        "character_accuracy_1_minus_cer": round(1.0 - cer, 6),
    }


def evaluate_rows(rows: list[dict], field: str) -> dict:
    return {
        "strict_nfkc_no_whitespace": evaluate_rows_with(rows, field, normalize_strict),
        "content_only_no_punctuation_or_symbols": evaluate_rows_with(rows, field, normalize_content),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))

    output = {"source": str(args.input), "per_image": {}, "combined": {}}
    all_rows: list[dict] = []
    total_gt_boxes = 0
    total_detected_boxes = 0
    for image in data["images"]:
        rows = image["aligned"]
        all_rows.extend(rows)
        total_gt_boxes += image["visual_gt_line_count"]
        total_detected_boxes += image["detected_box_count"]
        output["per_image"][image["image"]] = {
            "line_count_proxy": {
                "visual_gt_lines": image["visual_gt_line_count"],
                "detected_boxes": image["detected_box_count"],
                "count_ratio": round(image["detected_box_count"] / image["visual_gt_line_count"], 6),
                "warning": "仅为框数量一致性代理，不是带 polygon 标注的 Det precision/recall。",
            },
            "raw_recognizer": evaluate_rows(rows, "raw_rec"),
            "accepted_end_to_end": evaluate_rows(rows, "accepted"),
        }

    output["combined"] = {
        "line_count_proxy": {
            "visual_gt_lines": total_gt_boxes,
            "detected_boxes": total_detected_boxes,
            "count_ratio": round(total_detected_boxes / total_gt_boxes, 6),
            "warning": "仅为框数量一致性代理，不是带 polygon 标注的 Det precision/recall。",
        },
        "raw_recognizer": evaluate_rows(all_rows, "raw_rec"),
        "accepted_end_to_end": evaluate_rows(all_rows, "accepted"),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
