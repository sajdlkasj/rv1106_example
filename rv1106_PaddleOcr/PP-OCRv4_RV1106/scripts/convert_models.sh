#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
fi
conda activate toolkit2
mkdir -p "$ROOT/models/rknn"

DET_DIR="$ROOT/vendor/rknn_model_zoo/examples/PPOCR/PPOCR-Det/python"
(
  cd "$DET_DIR"
  python convert.py \
    "$ROOT/models/onnx/ppocrv4_det.onnx" \
    rv1106 i8 \
    "$ROOT/models/rknn/ppocrv4_det_rv1106_i8.rknn"
)

python "$ROOT/scripts/convert_rec_rv1106.py" \
  --model "$ROOT/models/onnx/ppocrv4_rec.onnx" \
  --dataset "$ROOT/vendor/rknn_model_zoo/datasets/PPOCR/imgs/dataset_20.txt" \
  --output "$ROOT/models/rknn/ppocrv4_rec_rv1106_i8.rknn"

printf '\nRKNN models are ready:\n  %s\n  %s\n' \
  "$ROOT/models/rknn/ppocrv4_det_rv1106_i8.rknn" \
  "$ROOT/models/rknn/ppocrv4_rec_rv1106_i8.rknn"
