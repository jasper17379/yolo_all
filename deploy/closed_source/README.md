# 闭源部署指南

本文档说明如何将 Vision AI Platform 以闭源方式部署到生产应用服务器。

## 1. 部署架构

```
                    ┌─────────────────┐
                    │  Nginx / 网关    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  FastAPI 服务    │
                    │  (Docker)       │
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌────▼────┐  ┌─────▼─────┐
        │ YOLO 权重  │  │ 人脸库   │  │ PaddleOCR │
        │ (.pt/.onnx)│  │ gallery │  │  模型     │
        └───────────┘  └─────────┘  └───────────┘
```

## 2. 代码保护方案

### 方案 A: PyInstaller 打包 (推荐中小规模)

```bash
pip install pyinstaller

# 打包 API 服务
pyinstaller --onefile --hidden-import=ultralytics \
  --add-data "configs;configs" \
  --add-data "weights;weights" \
  src/api/server.py -n vision_ai_server

# 产物在 dist/vision_ai_server，无需暴露 Python 源码
```

### 方案 B: Cython 编译核心模块

```bash
# 将 src/core, src/tasks 编译为 .so / .pyd
python setup_cython.py build_ext --inplace
# 部署时仅分发 .pyd/.so + 入口脚本
```

### 方案 C: Docker 镜像闭源

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY configs/ configs/
COPY weights/ weights/
COPY dist/vision_ai_server /app/server
EXPOSE 8000
CMD ["./server"]
```

## 3. 生产部署步骤

```bash
# 1. 导出 ONNX (可选，用于 C++ 侧推理)
python scripts/export_models.py --task all --format onnx

# 2. 构建 Docker 镜像
docker build -t vision-ai:1.0 -f deploy/Dockerfile .

# 3. 启动服务
docker run -d --gpus all -p 8000:8000 \
  -v /data/weights:/app/weights \
  -v /data/gallery:/app/datasets/face/gallery \
  vision-ai:1.0

# 4. 健康检查
curl http://localhost:8000/health
```

## 4. 模型与密钥管理

| 资产 | 存储位置 | 权限 |
|------|----------|------|
| 训练权重 | `/data/weights/` | 只读, 600 |
| 人脸特征库 | `/data/gallery/embeddings.pkl` | 600 |
| API 密钥 | 环境变量 `API_KEY` | 不写入镜像 |

## 5. 安全建议

1. **不要**将权重和 gallery 打入 Git 仓库
2. 使用 HTTPS + API Key 鉴权 (可在 `src/api/server.py` 添加 middleware)
3. 人脸数据符合 GDPR/个保法，需加密存储
4. 日志中不记录原始人脸图像

## 6. 更新模型流程

```bash
# 在训练机完成训练
python -m src.train.trainer --task helmet --resume-from-best --epochs 50

# 同步权重到生产
scp weights/helmet/best.pt user@server:/data/weights/helmet/

# 重启服务
docker restart vision-ai
```

## 7. 性能调优

- GPU: 设置 `device=cuda:0`，batch 推理
- CPU: 导出 ONNX + INT8 量化
- RK3588: 使用 RKNN 模型，NPU 推理 — 见 [../OFFLINE_WORKFLOW.md](../OFFLINE_WORKFLOW.md)
- x86 NVIDIA: C++ + ORT CUDA — 见 [../platforms/NVIDIA_X86_UBUNTU22.md](../platforms/NVIDIA_X86_UBUNTU22.md)
