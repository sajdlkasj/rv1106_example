#!/usr/bin/env python3
"""Build PP-OCRv4 Det for RV1106 while preserving a verbose placement log."""

import argparse
from pathlib import Path

from rknn.api import RKNN


def checked(step, result):
    if result != 0:
        raise SystemExit(f"{step} failed: {result}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(root / "models/onnx/ppocrv4_det.onnx"))
    parser.add_argument(
        "--dataset",
        default=str(root / "vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "models/rknn/candidates/ppocrv4_det_rv1106_i8_verbose.rknn"),
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        checked(
            "RKNN config",
            rknn.config(
                target_platform="rv1106",
                mean_values=[[123.675, 116.28, 103.53]],
                std_values=[[58.395, 57.12, 57.375]],
            ),
        )
        checked("ONNX load", rknn.load_onnx(model=args.model))
        checked("RKNN build", rknn.build(do_quantization=True, dataset=args.dataset))
        checked("RKNN export", rknn.export_rknn(str(output)))
    finally:
        rknn.release()
    print(f"SUCCESS: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
