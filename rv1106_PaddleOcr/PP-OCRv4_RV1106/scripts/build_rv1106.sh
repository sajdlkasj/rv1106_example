#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_TOOLCHAIN="$ROOT/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf"
TOOLCHAIN="${RV1106_TOOLCHAIN_PREFIX:-$DEFAULT_TOOLCHAIN}"
BUILD="$ROOT/build/rv1106_armhf"
INSTALL="$ROOT/install/rv1106_armhf/ppocr_rv1106"
if [[ ! -x "${TOOLCHAIN}-gcc" || ! -x "${TOOLCHAIN}-g++" ]]; then
  echo "RV1106 cross compiler was not found." >&2
  echo "Extract the Luckfox arm-rockchip830 toolchain under $ROOT/toolchain," >&2
  echo "or export RV1106_TOOLCHAIN_PREFIX=/absolute/path/to/arm-rockchip830-linux-uclibcgnueabihf" >&2
  exit 2
fi
export GCC_COMPILER="$TOOLCHAIN"
export PATH="$(dirname "$TOOLCHAIN"):$PATH"
"${TOOLCHAIN}-gcc" --version | head -1
cmake -S "$ROOT/src/ppocr_system" -B "$BUILD" \
  -DTARGET_SOC=rv1106 \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${TOOLCHAIN}-gcc" \
  -DCMAKE_CXX_COMPILER="${TOOLCHAIN}-g++" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL"
cmake --build "$BUILD" --parallel 4
cmake --install "$BUILD"
