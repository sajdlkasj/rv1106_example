project(wk_Gray_scale_matching)
cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
SET(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")


# === 2. 设置针对 RV1106 (Cortex-A7/A35) 的编译优化标志 ===
set(RV1106_COMPILE_FLAGS
    -mcpu=cortex-a7          # 或 cortex-a35，根据实际 CPU
    -mfpu=neon               # 启用 NEON SIMD
    -mfloat-abi=hard         # 使用硬件浮点（与你的 toolchain 匹配）
    -O3                      # 最大优化
    -funsafe-math-optimizations
    -ffast-math
    -fno-rtti
    -fno-exceptions
    -DNDEBUG
)


set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv-mobile-4.12.0-luckfox-pico/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ SRCS)


add_executable(wk_Gray_scale_matching main.cpp ${SRCS})

target_compile_options(wk_Gray_scale_matching PRIVATE  ${RV1106_COMPILE_FLAGS})

target_link_libraries(wk_Gray_scale_matching ${OpenCV_LIBS})