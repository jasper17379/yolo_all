# RK3588 + Ubuntu 22.04 — C++ 推理部署

目标：在 RK3588 板端 **离线运行** C++ 推理，YOLO 走 **NPU（RKNN）**，车牌/人脸走 CPU（ONNX Runtime 或 HyperLPR C++）。

## 推荐方案

| 模块 | 后端 | 库 |
|------|------|-----|
| YOLO 检测（helmet/plate/action） | **RKNN / NPU** | librknnrt |
| 车牌 OCR | HyperLPR C++ 或 ORT CPU | libhyperlpr3 / onnxruntime |
| 人脸 | ORT CPU | onnxruntime + buffalo_l ONNX |
| 图像 | OpenCV | libopencv |

## 一、PC 端（x86 Ubuntu 22.04，联网一次）

### 1.1 安装转换工具

```bash
# rknn-toolkit2 仅用于 PT/ONNX → RKNN，不能装在 RK3588 上
pip install rknn-toolkit2 onnx

# 本项目训练/导出环境
pip install -r requirements.txt   # torch/onnxruntime 等通用库
python scripts/setup_third_party.py
python scripts/setup_yolo_sources.py
```

### 1.2 训练 & 导出

```bash
python -m src.train.trainer --task helmet --device 0 --epochs 50
python scripts/export_models.py --task helmet --format onnx --yolo yolov8 --model-size n
python scripts/export_models.py --task helmet --format rknn --yolo yolov8 --model-size n
```

INT8 量化建议准备 `datasets/helmet/images/val` 下 50+ 张图作为校准集（Ultralytics export 会自动采样）。

### 1.3 交叉编译 C++（在 PC 上编 aarch64）

```bash
# 依赖: aarch64 交叉工具链 + RK3588 sysroot（含 librknnrt、OpenCV）
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

cd deploy/cpp
mkdir build-rk3588 && cd build-rk3588

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchain.aarch64-rk3588.cmake \
  -DRK3588_TOOLCHAIN_ROOT=/usr \
  -DVISION_BACKEND=RKNN \
  -DRKNN_RT_DIR=/path/to/rknpu2/runtime/RK3588/Linux/librknn_api/aarch64

cmake --build . -j$(nproc)
# 产物: vision_infer_rknn
```

### 1.4 板端本地编译（可选，板子性能足够时）

```bash
# 在 RK3588 本机
sudo apt install build-essential cmake libopencv-dev

cd deploy/cpp && mkdir build && cd build
cmake .. \
  -DVISION_BACKEND=RKNN \
  -DRKNN_RT_DIR=/usr/lib/rknn
cmake --build . -j$(nproc)
```

---

## 二、RK3588 板端运行时库

### 2.1 必需（方案 A：RKNN）

| 库 | 来源 | 说明 |
|----|------|------|
| **librknnrt.so** | [rknpu2](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2) | NPU 运行时 |
| **libopencv_*.so** | `apt install libopencv-dev` 或拷贝 | 图像读写/预处理 |
| **libstdc++.so** | 系统自带 | C++17 |

### 2.2 可选模块

| 库 | 启用 CMake 选项 | 用途 |
|----|-----------------|------|
| librga.so | `VISION_USE_RGA=ON` | 硬件 resize |
| libhyperlpr3.so | `VISION_ENABLE_PLATE_HYPERLPR=ON` | 车牌 C++ |
| libonnxruntime.so | `VISION_BACKEND=ORT_CPU` | 调试 / 人脸 ONNX |

HyperLPR C++ 编译（PC 或板端）：

```bash
cd third_party/HyperLPR
mkdir build && cd build
cmake .. -DBUILD_SHARE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
# install 到 deploy 指定 HYPERLPR_INSTALL_DIR
```

### 2.3 不需要（C++ RKNN 方案）

- Python / pip / torch / ultralytics
- CUDA

---

## 三、离线拷贝清单

```
/opt/vision_ai/
├── bin/vision_infer_rknn
├── models/
│   ├── helmet_yolov8n.rknn
│   ├── plate_yolov8n.rknn
│   ├── action_yolov8n.rknn
│   ├── hyperlpr3/20230229/onnx/     # 若用 ORT 做 OCR
│   └── insightface/models/buffalo_l/ # 人脸 det_10g.onnx 等
├── lib/                             # 若板端无系统包
│   └── librknnrt.so
└── configs/classes.yaml
```

从本项目拷贝模型路径：

- YOLO RKNN：export 产出或 `weights/{task}/`
- HyperLPR：`third_party/models/hyperlpr3/`
- InsightFace：`third_party/models/insightface/models/buffalo_l/`

---

## 四、运行

```bash
export LD_LIBRARY_PATH=/opt/vision_ai/lib:$LD_LIBRARY_PATH

./vision_infer_rknn --model models/helmet_yolov8n.rknn --source test.jpg
# 视频流（实现 RTSP 捕获后）:
# ./vision_infer_rknn --model models/helmet_yolov8n.rknn --source rtsp://...
```

---

## 五、CMake 选项速查

```bash
cmake .. \
  -DVISION_BACKEND=RKNN \
  -DRKNN_RT_DIR=/usr/lib/rknn \
  -DRKNN_INCLUDE_DIR=/usr/include/rknn \
  # -DVISION_USE_RGA=ON -DRGA_DIR=/usr/lib/rga \
  # -DVISION_ENABLE_PLATE_HYPERLPR=ON -DHYPERLPR_INSTALL_DIR=/opt/hyperlpr3 \
  # -DVISION_ENABLE_FACE_ONNX=ON -DINSIGHTFACE_MODEL_DIR=../../third_party/models/insightface/models/buffalo_l
```

完整选项说明见 [../cmake/Options.cmake](../cmake/Options.cmake)。

---

## 六、性能参考（调优方向）

- YOLO 输入尺寸：`--imgsz 640` → RKNN 可试 416 提升 FPS
- INT8 量化：必备，FP16/FP32 慢且占带宽
- 多模型：helmet + plate 可串行或按帧轮换，避免 NPU 并发冲突
- 人脸放 CPU ORT，与 NPU YOLO 流水线并行

---

## 七、调试回退

若 RKNN 转换有问题，可临时用 **ORT CPU** 在板端验证逻辑：

```bash
cmake .. -DVISION_BACKEND=ORT_CPU \
  -DONNXRUNTIME_DIR=/opt/onnxruntime-aarch64
# 需 aarch64 版 ONNX Runtime: onnxruntime-linux-aarch64-*.tgz
```
