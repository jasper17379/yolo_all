# Vision AI C++ 推理工程

YOLO 检测 C++ 骨架，支持 **ONNX Runtime**（x86 / RK3588 CPU）与 **RKNN**（RK3588 NPU）两种后端。

## 文档

- 总览：[../README.md](../README.md)
- 离线流程 + 方案 A：[../OFFLINE_WORKFLOW.md](../OFFLINE_WORKFLOW.md)
- RK3588：[../platforms/RK3588_UBUNTU22.md](../platforms/RK3588_UBUNTU22.md)
- NVIDIA x86：[../platforms/NVIDIA_X86_UBUNTU22.md](../platforms/NVIDIA_X86_UBUNTU22.md)
- CMake 选项：[../cmake/Options.cmake](../cmake/Options.cmake)

## 目录

```
deploy/cpp/
├── CMakeLists.txt
├── include/yolo_detector.h
└── src/
    ├── main.cpp
    └── yolo_detector.cpp    # ONNX 骨架；RKNN 待 yolo_detector_rknn.cpp
```

## 快速编译

### NVIDIA + Ubuntu 22.04（CUDA）

```bash
cd deploy/cpp && mkdir build && cd build
cmake .. -DVISION_BACKEND=ORT_CUDA -DONNXRUNTIME_DIR=/opt/onnxruntime
cmake --build . -j$(nproc)
./vision_infer --model ../../../weights/helmet/best_yolov8n.onnx --source test.jpg
```

### RK3588 + Ubuntu 22.04（RKNN）

```bash
cmake .. -DVISION_BACKEND=RKNN -DRKNN_RT_DIR=/usr/lib/rknn
cmake --build . -j$(nproc)
./vision_infer_rknn --model helmet.rknn --source test.jpg
```

### RK3588 交叉编译

```bash
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchain.aarch64-rk3588.cmake \
  -DVISION_BACKEND=RKNN \
  -DRKNN_RT_DIR=/path/to/rknpu2/runtime
```

## 模型导出（PC Python）

```bash
python scripts/export_models.py --task helmet --format onnx
python scripts/export_models.py --task helmet --format rknn   # RK3588
```

## 实现状态

| 组件 | 状态 |
|------|------|
| CMake 多后端选项 | 已完成 |
| OpenCV 预处理/画框 | 骨架 |
| ONNX Runtime Session | TODO（见 yolo_detector.cpp） |
| RKNN 推理 | TODO（参考 rknn_model_zoo） |
| HyperLPR / InsightFace 模块 | CMake 预留选项 |

## 环境变量

| 变量 | 说明 |
|------|------|
| `ONNXRUNTIME_DIR` | ONNX Runtime 安装路径 |
| `CUDA_PATH` | CUDA 路径（ORT_CUDA） |
| `LD_LIBRARY_PATH` | 需含 onnxruntime / rknnrt |
