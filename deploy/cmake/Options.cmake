# deploy/cmake/Options.cmake — Vision AI C++ 推理构建选项

# ---------------------------------------------------------------------------
# 目标平台（二选一，见 deploy/platforms/ 文档）
#   ORT_CPU   — x86/aarch64 + ONNX Runtime CPU（Intel CPU / RK3588 调试）
#   ORT_CUDA  — x86 + ONNX Runtime CUDA EP（NVIDIA GPU）
#   RKNN      — RK3588 NPU（量产推荐）
# ---------------------------------------------------------------------------
set(VISION_BACKEND "ORT_CPU" CACHE STRING "ORT_CPU | ORT_CUDA | RKNN")
set_property(CACHE VISION_BACKEND PROPERTY STRINGS ORT_CPU ORT_CUDA RKNN)

# ---------------------------------------------------------------------------
# ONNX Runtime（ORT_CPU / ORT_CUDA 必选其一）
# 下载: https://github.com/microsoft/onnxruntime/releases
#   x86_64: onnxruntime-linux-x64-*.tgz
#   CUDA:   onnxruntime-linux-x64-gpu-*.tgz
#   aarch64(RK3588): onnxruntime-linux-aarch64-*.tgz
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_DIR "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime 根目录 (含 include/ lib/)")

# CUDA EP 时指定 CUDA 路径（Ubuntu 22.04 + NVIDIA 驱动已装）
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA toolkit (ORT_CUDA)")

# ---------------------------------------------------------------------------
# RKNN（VISION_BACKEND=RKNN 时启用）
# 来源: https://github.com/airockchip/rknn-toolkit2
# 板端 runtime: librknnrt.so (rknpu2)
# PC 端转换: rknn-toolkit2（仅 x86，不可装在 RK3588 上）
# ---------------------------------------------------------------------------
set(RKNN_RT_DIR "/usr/lib/rknn" CACHE PATH "RKNN runtime 目录 (含 librknnrt.so)")
set(RKNN_INCLUDE_DIR "${RKNN_RT_DIR}/include" CACHE PATH "RKNN 头文件目录")

# Rockchip RGA 硬件缩放（可选，RK3588 预处理加速）
# set(RGA_DIR "/usr/lib/rga" CACHE PATH "librga 目录")
# set(VISION_USE_RGA OFF CACHE BOOL "使用 RGA 做 resize/格式转换")

# ---------------------------------------------------------------------------
# 业务模块（可选，后续接入时打开）
# ---------------------------------------------------------------------------

# 车牌 OCR — HyperLPR3 C++ SDK（third_party/HyperLPR/cpp）
# 编译 HyperLPR 后设置:
#   HYPERLPR_INSTALL_DIR=/path/to/hyperlpr3/install
# set(HYPERLPR_INSTALL_DIR "" CACHE PATH "HyperLPR3 C++ install prefix")
# set(VISION_ENABLE_PLATE_HYPERLPR OFF CACHE BOOL "启用车牌 HyperLPR3 C++ 模块")

# 人脸 — InsightFace ONNX（scrfd + w600k_r50，模型在 third_party/models/insightface/）
# C++ 侧直接用 ONNX Runtime 加载 .onnx，无需 Python insightface 包
# set(VISION_ENABLE_FACE_ONNX OFF CACHE BOOL "启用人脸 ONNX 推理模块")
# set(INSIGHTFACE_MODEL_DIR "${CMAKE_SOURCE_DIR}/../../third_party/models/insightface/models/buffalo_l")

# PaddleOCR C++（较重，RK3588 不推荐；x86 可选）
# set(VISION_ENABLE_PADDLE_OCR OFF CACHE BOOL "启用 PaddleOCR C++ 模块")

# ---------------------------------------------------------------------------
# 交叉编译（RK3588 在 x86 主机上编译时）
# 用法: cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchain.aarch64-rk3588.cmake ..
# ---------------------------------------------------------------------------
# set(RK3588_TOOLCHAIN_ROOT "/opt/rk3588/aarch64-linux-gnu" CACHE PATH "aarch64 交叉工具链根目录")
