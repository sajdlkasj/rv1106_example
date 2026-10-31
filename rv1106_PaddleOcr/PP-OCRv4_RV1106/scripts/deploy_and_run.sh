#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL="$ROOT/install/rv1106_armhf/ppocr_rv1106"
DEVICE_DIR="/tmp/ppocr_rv1106"
IMAGE="${1:-}"
ADB=(sudo adb)
if [[ "$EUID" -eq 0 ]]; then ADB=(adb); fi

if [[ ! -x "$INSTALL/rknn_ppocr_system_demo" ]]; then
  echo "Demo is not built. Run scripts/build_rv1106.sh first." >&2
  exit 1
fi
"${ADB[@]}" get-state >/dev/null
"${ADB[@]}" shell "rm -rf $DEVICE_DIR"
"${ADB[@]}" push "$INSTALL" "$DEVICE_DIR" >/dev/null

BOARD_IMAGE="model/test.jpg"
if [[ -n "$IMAGE" ]]; then
  if [[ ! -f "$IMAGE" ]]; then
    echo "Image not found: $IMAGE" >&2
    exit 1
  fi
  "${ADB[@]}" push "$IMAGE" "$DEVICE_DIR/model/input.jpg" >/dev/null
  BOARD_IMAGE="model/input.jpg"
fi

mkdir -p "$ROOT/output" "$ROOT/logs"
RUN_LOG="$ROOT/logs/rv1106_run.log"
RESULT_IMAGE="$ROOT/output/rv1106_test_result.jpg"
"${ADB[@]}" shell "cd $DEVICE_DIR && export LD_LIBRARY_PATH=$DEVICE_DIR/lib && ./rknn_ppocr_system_demo model/ppocrv4_det_i8.rknn model/ppocrv4_rec_i8.rknn $BOARD_IMAGE" 2>&1 | tee "$RUN_LOG"
rm -f "$RESULT_IMAGE"
: > "$RESULT_IMAGE"
"${ADB[@]}" pull "$DEVICE_DIR/out.jpg" "$RESULT_IMAGE" >/dev/null
TEMP_RESULT="$RESULT_IMAGE.user"
cp "$RESULT_IMAGE" "$TEMP_RESULT"
mv -f "$TEMP_RESULT" "$RESULT_IMAGE"
printf '\nResult image: %s\nRun log: %s\n' "$RESULT_IMAGE" "$RUN_LOG"
