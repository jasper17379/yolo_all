cmake_minimum_required(VERSION 3.14)
project(vision_ai_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# ONNX Runtime - 用户需指定路径
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime install dir")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(vision_infer
    src/main.cpp
    src/yolo_detector.cpp
)

target_link_libraries(vision_infer
    ${OpenCV_LIBS}
    onnxruntime
)

install(TARGETS vision_infer DESTINATION bin)
