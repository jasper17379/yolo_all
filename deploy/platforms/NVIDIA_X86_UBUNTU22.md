# NVIDIA GPU + Intel CPU + Ubuntu 22.04 — C++ 推理部署

目标：在 x86_64 工控机/服务器上 **C++ 低延迟推理**，YOLO 走 **ONNX Runtime + CUDA EP**，同样推荐 C++ 而非 Python 全栈服务。

## 推荐方案

| 模块 | 后端 | 库 |
|------|------|-----|
| YOLO 检测 | **ONNX Runtime CUDA** | onnxruntime (GPU) + CUDA 12.x |
| 车牌 OCR | HyperLPR3 ORT / HyperLPR C++ | onnxruntime / libhyperlpr3 |
| 人脸 | ORT CUDA 或 CPU | buffalo_l ONNX |
| 图像 | OpenCV | libopencv |

**同样推荐 C++ 的原因**：比 `demo.py` / FastAPI Python 延迟低、无 GIL、生产环境依赖可控（无需在服务器上装 torch）。

---

## 一、环境准备（Ubuntu 22.04）

### 1.1 系统依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libopencv-dev \
  nvidia-driver-535    # 按 GPU 型号选择
```

### 1.2 CUDA（GPU 推理）

```bash
# Ubuntu 22.04 推荐 CUDA 12.x（与 onnxruntime-gpu 版本匹配）
# 安装后验证
nvidia-smi
nvcc --version
```

### 1.3 ONNX Runtime（GPU 版）

从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 下载：

```
onnxruntime-linux-x64-gpu-1.19.x.tgz
```

解压到 `/opt/onnxruntime`：

```
/opt/onnxruntime/
├── include/
├── lib/
│   └── libonnxruntime.so
└── ...
```

CPU-only 机器用 `onnxruntime-linux-x64-*.tgz`（无 gpu 后缀）。

### 1.4 本项目 Python 环境（仅 PC 训练/导出，板端不需要）

```bash
pip install -r requirements.txt
python scripts/setup_third_party.py
python scripts/setup_yolo_sources.py
python scripts/check_offline_deps.py
```

---

## 二、训练 & 导出（PC）

```bash
# 训练
python -m src.train.trainer --task helmet --device 0 --epochs 50

# 导出 ONNX（C++ 直接使用）
python scripts/export_models.py --task all --format onnx --yolo yolov8 --model-size n --device 0

# 可选 TensorRT（NVIDIA 极致性能）
python scripts/export_models.py --task helmet --format engine --device 0
```

产物示例：

- `weights/helmet/best_yolov8n.onnx`
- `weights/helmet/best_yolov8n.engine`（TensorRT，需 GPU 导出）

---

## 三、编译 C++ 推理程序

### 3.1 GPU 版（推荐）

```bash
cd deploy/cpp
mkdir build && cd build

cmake .. \
  -DVISION_BACKEND=ORT_CUDA \
  -DONNXRUNTIME_DIR=/opt/onnxruntime \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

cmake --build . -j$(nproc)

# 产物: vision_infer
```

运行前确保 `LD_LIBRARY_PATH` 包含 ONNX Runtime 与 CUDA：

```bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
./vision_infer --model ../../weights/helmet/best_yolov8n.onnx --source test.jpg
```

### 3.2 CPU 版（无 GPU 或轻负载）

```bash
cmake .. \
  -DVISION_BACKEND=ORT_CPU \
  -DONNXRUNTIME_DIR=/opt/onnxruntime
cmake --build . -j$(nproc)
```

Intel CPU 上可配合 OpenVINO 作为后续优化方向（当前 CMake 未集成，预留 ONNX 统一路径）。

---

## 四、运行时库清单

### 4.1 必需

| 库 | 说明 |
|----|------|
| libonnxruntime.so | ONNX Runtime |
| libopencv_core/imgproc/highgui.so | OpenCV |
| libcudart.so / libcublas.so 等 | 仅 ORT_CUDA |
| libnvinfer.so | 仅 TensorRT engine 方案 |

### 4.2 可选业务模块

| 模块 | 模型路径（项目内） | CMake 选项 |
|------|-------------------|------------|
| 车牌 HyperLPR C++ | 编译 `third_party/HyperLPR/cpp` | `VISION_ENABLE_PLATE_HYPERLPR` |
| 车牌 OCR Python 模型 | `third_party/models/hyperlpr3/` | C++ 用 ORT 加载 ONNX |
| 人脸 InsightFace | `third_party/models/insightface/models/buffalo_l/` | `VISION_ENABLE_FACE_ONNX` |

InsightFace **Python 源码**：`third_party/insightface/python-package`（训练/录入 gallery 用）  
InsightFace **C++ 推理**：直接加载 `det_10g.onnx`、`w600k_r50.onnx` 等，无需 Python。

### 4.3 不需要（C++ 部署）

- pip install ultralytics / insightface / hyperlpr3（均已 vendored 到 third_party，仅 Python 工具链需要）
- 板端若纯 C++ 推理：**不需要 Python**

---

## 五、离线部署清单

拷贝到无网服务器：

```
/opt/vision_ai/
├── bin/vision_infer
├── models/
│   ├── helmet_yolov8n.onnx
│   ├── plate_yolov8n.onnx
│   ├── action_yolov8n.onnx
│   ├── hyperlpr3/
│   └── insightface/models/buffalo_l/
├── lib/                    # 若目标机未 apt 安装
│   ├── libonnxruntime.so
│   └── ...
└── configs/
```

离线服务器仍需安装：**NVIDIA 驱动 + CUDA runtime**（若用 GPU）。

---

## 六、与 Python 服务的关系

| 场景 | 方案 |
|------|------|
| 生产低延迟视频分析 | **C++ vision_infer**（本文） |
| REST API / 快速迭代 | Python `src/api/server.py` 或 Docker |
| 闭源交付 | [../closed_source/README.md](../closed_source/README.md) |

可在同一机器上 C++ 做实时流，Python API 做管理/录入人脸。

---

## 七、CMake 选项速查

```bash
cmake .. \
  -DVISION_BACKEND=ORT_CUDA \
  -DONNXRUNTIME_DIR=/opt/onnxruntime \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  # -DVISION_ENABLE_PLATE_HYPERLPR=ON -DHYPERLPR_INSTALL_DIR=/opt/hyperlpr3 \
  # -DVISION_ENABLE_FACE_ONNX=ON \
  #   -DINSIGHTFACE_MODEL_DIR=${PROJECT}/third_party/models/insightface/models/buffalo_l
```

详见 [../cmake/Options.cmake](../cmake/Options.cmake)。

---

## 八、性能调优

- ORT CUDA EP：Session 创建时启用 `CUDAExecutionProvider`
- FP16：`export` 时 `half=True`（需 GPU）
- TensorRT：`--format engine` 进一步降低延迟
- 批处理：多路摄像头时可 batch infer（需改 C++ 主循环）
