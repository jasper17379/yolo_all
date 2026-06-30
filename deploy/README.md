# Vision AI 部署指南

本目录包含 **C++ 边缘推理**、**离线落地流程**、**双平台编译方案** 及闭源部署说明。

## 文档索引

| 文档 | 内容 |
|------|------|
| [OFFLINE_WORKFLOW.md](OFFLINE_WORKFLOW.md) | 离线落地总流程 + **方案 A（C++ + RKNN 量产）** |
| [platforms/RK3588_UBUNTU22.md](platforms/RK3588_UBUNTU22.md) | RK3588 + Ubuntu 22.04：库清单、编译、拷贝 |
| [platforms/NVIDIA_X86_UBUNTU22.md](platforms/NVIDIA_X86_UBUNTU22.md) | NVIDIA GPU + Intel CPU + Ubuntu 22.04 |
| [cpp/README.md](cpp/README.md) | C++ 工程结构与 CMake 快速入门 |
| [closed_source/README.md](closed_source/README.md) | Docker / PyInstaller 闭源 Python 服务 |

## 推荐部署方式对比

| 硬件 | 推荐方案 | 推理后端 | 说明 |
|------|----------|----------|------|
| **RK3588 + Ubuntu 22** | **C++ + RKNN** | NPU | 量产首选，低功耗高 FPS |
| **RK3588 + Ubuntu 22** | C++ + ORT CPU | CPU | 调试/无 RKNN 转换时 |
| **NVIDIA + Intel + Ubuntu 22** | **C++ + ORT CUDA** | GPU | 服务器/工控机首选 |
| **NVIDIA + Intel + Ubuntu 22** | C++ + ORT CPU | CPU | 无 GPU 或轻负载 |
| 任意（原型） | Python 全栈 | torch/ort | 见 `scripts/check_offline_deps.py` |

**NVIDIA + Intel + Ubuntu 22 同样推荐 C++ 部署**：GPU 用 ONNX Runtime CUDA EP，比 Python 全栈延迟更低、依赖更可控。

## 目录结构

```
deploy/
├── README.md                 # 本文件
├── OFFLINE_WORKFLOW.md       # 离线流程 + 方案 A
├── platforms/
│   ├── RK3588_UBUNTU22.md
│   └── NVIDIA_X86_UBUNTU22.md
├── cmake/
│   ├── Options.cmake         # 构建选项（ORT/RKNN/可选模块）
│   └── Toolchain.aarch64-rk3588.cmake
├── cpp/                      # C++ 推理工程
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── Dockerfile                # Python API 容器（非 C++ 方案）
└── closed_source/
```

## 模型与 third_party 对应

| 任务 | 训练产出 | PC 导出 | 板端 C++ 使用 |
|------|----------|---------|---------------|
| 安全帽/动作/车牌检测 | `weights/{task}/best_yolov8n.pt` | `.onnx` → `.rknn` | RKNN 或 ONNX |
| 车牌读字 | — | HyperLPR3 ONNX | HyperLPR C++ 或 ORT |
| 人脸 | gallery + buffalo_l | InsightFace ONNX | ORT（`det_10g.onnx` 等） |

Python 业务源码均已 vendored 至 `third_party/`（ultralytics、HyperLPR、insightface），见项目根 `third_party/README.md`。

## 快速开始

```bash
# 1. PC 端导出（联网一次）
python scripts/export_models.py --task helmet --format onnx
python scripts/export_models.py --task helmet --format rknn   # 需 rknn-toolkit2

# 2. 编译 C++（按平台见 platforms/ 文档）
cd deploy/cpp && mkdir build && cd build
cmake .. -DVISION_BACKEND=ORT_CUDA -DONNXRUNTIME_DIR=/opt/onnxruntime
cmake --build . -j$(nproc)
```
