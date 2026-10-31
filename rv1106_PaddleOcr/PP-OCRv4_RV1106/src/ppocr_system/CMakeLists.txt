cmake_minimum_required(VERSION 3.10)

project(rknn_ppocr_system_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

get_filename_component(MZ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/rknn_model_zoo" ABSOLUTE)
add_subdirectory("${MZ_ROOT}/3rdparty" 3rdparty.out)
add_subdirectory("${MZ_ROOT}/utils" utils.out)

#opencv
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
message(STATUS "64bit")
set (TARGET_LIB_ARCH lib64)
else()
message(STATUS "32bit")
set (TARGET_LIB_ARCH lib)
endif()
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(OpenCV_DIR ${MZ_ROOT}/3rdparty/opencv/opencv-android-sdk-build/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
    if (TARGET_SOC STREQUAL "x86-64")
      # set(OpenCV_DIR ${MZ_ROOT}/libs/common/opencv/opencv-linux-x86_64/share/OpenCV)
    elseif(TARGET_LIB_ARCH STREQUAL "lib")
      set(OpenCV_DIR ${MZ_ROOT}/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
    else()
      set(OpenCV_DIR ${MZ_ROOT}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
    endif()
  endif()
find_package(OpenCV REQUIRED)
message(STATUS OpenCV_DIR=${OpenCV_DIR})
message(STATUS OpenCV_LIBS=${OpenCV_LIBS})

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

if (TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    set(ppocr_system_file rknpu1/ppocr_system.cc)
else()
    set(ppocr_system_file rknpu2/ppocr_system.cc)
endif()


file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME}
    main.cc
    postprocess.cc
    clipper.cc
    ${ppocr_system_file}
)

target_link_libraries(${PROJECT_NAME}
    fileutils
    imageutils
    imagedrawing
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} log )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()


target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../data/test.jpg" DESTINATION model)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../models/rknn/ppocrv4_det_rv1106_i8.rknn" DESTINATION model RENAME ppocrv4_det_i8.rknn)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../models/rknn/ppocrv4_rec_rv1106_i8.rknn" DESTINATION model RENAME ppocrv4_rec_i8.rknn)
