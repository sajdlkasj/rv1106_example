project(rv1106_OCR)
cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)

#SET(CMAKE_C_COMPILER "/home/lubancat/Desktop/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
#SET(CMAKE_CXX_COMPILER "/home/lubancat/Desktop/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv-mobile-4.12.0-luckfox-pico/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include)

aux_source_directory(${CMAKE_SOURCE_DIR}/src SRCS)
link_directories(${CMAKE_SOURCE_DIR}/ncnn/lib)




add_executable(rv1106_OCR main.cpp ${SRCS})

target_link_libraries(rv1106_OCR ${OpenCV_LIBS} ncnn)

install(TARGETS rv1106_OCR DESTINATION ${CMAKE_SOURCE_DIR}/build/bin)

install(DIRECTORY   ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_SOURCE_DIR}/build/bin)
install(DIRECTORY   ${CMAKE_SOURCE_DIR}/image DESTINATION ${CMAKE_SOURCE_DIR}/build/bin)