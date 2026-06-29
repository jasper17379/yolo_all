# Vision AI C++ 推理部署

基于 ONNX Runtime 的 YOLO 检测推理，支持:
- Ubuntu + NVIDIA GPU (CUDA Execution Provider)
- RK3588 (RKNN / ONNX Runtime CPU)

## 目录结构

```
deploy/cpp/
├── CMakeLists.txt
├── include/yolo_detector.h
├── src/yolo_detector.cpp
├── src/main.cpp
└── README.md
```

## 构建步骤 (Ubuntu + NVIDIA)

```bash
# 1. 导出 ONNX 模型 (Python)
python -m src.infer.inferencer --task helmet --source datasets/helmet/images/val  # 先确保有权重
python scripts/export_models.py --task helmet --format onnx

# 2. 安装依赖
sudo apt install -y build-essential cmake libopencv-dev
# 下载 ONNX Runtime: https://github.com/microsoft/onnxruntime/releases

# 3. 编译
cd deploy/cpp
mkdir build && cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build . -j$(nproc)

# 4. 运行
./vision_infer --model ../../weights/helmet/best.onnx --source test.jpg
```

## RK3588 部署

```bash
# 1. 导出 RKNN 格式 (在 x86 主机上使用 rknn-toolkit2)
python scripts/export_models.py --task helmet --format rknn

# 2. 交叉编译或使用 RK3588 本地编译
# 参考 Rockchip RKNN SDK: https://github.com/airockchip/rknn-toolkit2

# 3. 在板端运行
./vision_infer_rknn --model best.rknn --source test.jpg
```

## 环境变量

| 变量 | 说明 |
|------|------|
| ONNXRUNTIME_DIR | ONNX Runtime 安装路径 |
| CUDA_PATH | CUDA 路径 (GPU 推理) |
