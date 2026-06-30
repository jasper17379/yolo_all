cmake_minimum_required(VERSION 3.14)
project(vision_ai_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 构建选项见 deploy/cmake/Options.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(Options)

find_package(OpenCV REQUIRED)

# ===========================================================================
# 源文件
# ===========================================================================
set(VISION_SOURCES
    src/main.cpp
    src/yolo_detector.cpp
)

# ===========================================================================
# ONNX Runtime 后端 (VISION_BACKEND = ORT_CPU | ORT_CUDA)
# ===========================================================================
if(VISION_BACKEND STREQUAL "ORT_CPU" OR VISION_BACKEND STREQUAL "ORT_CUDA")
    if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR
            "未找到 ONNX Runtime: ${ONNXRUNTIME_DIR}\n"
            "请下载并设置 -DONNXRUNTIME_DIR=...\n"
            "见 deploy/platforms/NVIDIA_X86_UBUNTU22.md 或 RK3588_UBUNTU22.md")
    endif()

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    link_directories(${ONNXRUNTIME_DIR}/lib)

    add_executable(vision_infer ${VISION_SOURCES})
    target_compile_definitions(vision_infer PRIVATE VISION_BACKEND_ORT=1)
    target_link_libraries(vision_infer ${OpenCV_LIBS} onnxruntime)

    if(VISION_BACKEND STREQUAL "ORT_CUDA")
        target_compile_definitions(vision_infer PRIVATE VISION_ORT_CUDA=1)
        # ORT CUDA EP 需链接 CUDA 库；Session 创建时指定 CUDAExecutionProvider
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            target_include_directories(vision_infer PRIVATE ${CUDA_INCLUDE_DIRS})
            target_link_libraries(vision_infer ${CUDA_LIBRARIES})
        else()
            message(WARNING "VISION_BACKEND=ORT_CUDA 但未找到 CUDA，将仅 CPU EP")
        endif()
    endif()

    message(STATUS "Backend: ONNX Runtime (${VISION_BACKEND})")
    message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

# ===========================================================================
# RKNN 后端 (VISION_BACKEND = RKNN) — RK3588 NPU
# ===========================================================================
elseif(VISION_BACKEND STREQUAL "RKNN")
    if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
        message(FATAL_ERROR
            "未找到 RKNN SDK: ${RKNN_INCLUDE_DIR}\n"
            "请安装 rknpu2 runtime 并设置 -DRKNN_RT_DIR=...\n"
            "见 deploy/platforms/RK3588_UBUNTU22.md")
    endif()

    # TODO: 实现 src/yolo_detector_rknn.cpp 并替换 yolo_detector.cpp
    # 参考: https://github.com/airockchip/rknn_model_zoo (examples/yolov8)

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${RKNN_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )
    link_directories(${RKNN_RT_DIR})

    add_executable(vision_infer_rknn ${VISION_SOURCES})
    target_compile_definitions(vision_infer_rknn PRIVATE VISION_BACKEND_RKNN=1)
    target_link_libraries(vision_infer_rknn ${OpenCV_LIBS} rknnrt)

    # 可选: RGA 硬件预处理
    # if(VISION_USE_RGA AND EXISTS "${RGA_DIR}/lib/librga.so")
    #     target_compile_definitions(vision_infer_rknn PRIVATE VISION_USE_RGA=1)
    #     target_include_directories(vision_infer_rknn PRIVATE ${RGA_DIR}/include)
    #     target_link_libraries(vision_infer_rknn rga)
    # endif()

    message(STATUS "Backend: RKNN (RK3588 NPU)")
    message(STATUS "RKNN_RT_DIR: ${RKNN_RT_DIR}")

else()
    message(FATAL_ERROR "未知 VISION_BACKEND: ${VISION_BACKEND}")
endif()

# ===========================================================================
# 可选业务模块（预留，默认 OFF）
# ===========================================================================

# --- HyperLPR3 C++ 车牌识别 ---
# if(VISION_ENABLE_PLATE_HYPERLPR AND HYPERLPR_INSTALL_DIR)
#     target_include_directories(vision_infer PRIVATE ${HYPERLPR_INSTALL_DIR}/include)
#     target_link_libraries(vision_infer ${HYPERLPR_INSTALL_DIR}/lib/libhyperlpr3.so)
#     target_compile_definitions(vision_infer PRIVATE VISION_ENABLE_PLATE=1)
# endif()

# --- InsightFace ONNX 人脸（buffalo_l 的 det_10g.onnx + w600k_r50.onnx）---
# if(VISION_ENABLE_FACE_ONNX)
#     target_compile_definitions(vision_infer PRIVATE VISION_ENABLE_FACE=1)
#     # 推理逻辑: ONNX Runtime 加载 INSIGHTFACE_MODEL_DIR 下 onnx
# endif()

if(VISION_BACKEND STREQUAL "RKNN")
    install(TARGETS vision_infer_rknn DESTINATION bin)
else()
    install(TARGETS vision_infer DESTINATION bin)
endif()
