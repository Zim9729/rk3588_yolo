cmake_minimum_required(VERSION 3.16)
project(rk3588_two_stage_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# aarch64 static OpenCV bundled with rknn_model_zoo; override with -DOpenCV_DIR=...
set(OpenCV_DIR "/home/hzx/hhm/rknn_model_zoo-main/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV"
    CACHE PATH "Directory containing OpenCVConfig.cmake")
find_package(OpenCV REQUIRED core imgproc imgcodecs)
find_library(RKNNRT_LIB NAMES rknnrt REQUIRED)

add_executable(two_stage_demo
    src/main.cpp
    src/rknn_detector.cpp
    src/yolo.cpp)
target_include_directories(two_stage_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknn/include
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(two_stage_demo PRIVATE ${OpenCV_LIBS} ${RKNNRT_LIB})
