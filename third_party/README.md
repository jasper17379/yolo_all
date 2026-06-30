# third_party — 业务依赖源码与模型

本目录集中存放 **检测框架 + 业务场景** 的源码与模型权重，避免依赖散落在 `C:\Python312\...` 或用户家目录。

## 目录结构

```
third_party/
├── yolov5/              # YOLOv5 源码
├── ultralytics/         # YOLOv8 / YOLOv10 源码
├── insightface/         # InsightFace 源码（人脸识别）
├── PaddleOCR/           # PaddleOCR 源码（车牌 OCR）
└── models/              # 模型权重与缓存（运行时实际加载）
    ├── insightface/
    │   └── models/
    │       └── buffalo_l/    # 人脸检测+特征 ONNX
    └── paddleocr/            # PaddleOCR / PaddleX 下载的 OCR 模型
```

## 一键初始化

```bash
# 克隆全部源码 + 迁移/下载人脸模型 + YOLO 源码
python scripts/setup_third_party.py

# 仅整理模型（不 git clone）
python scripts/setup_third_party.py --skip-clone --download-face-model
```

若 `~/.insightface/models/buffalo_l` 已存在，会自动 **复制** 到 `third_party/models/insightface/models/buffalo_l`。

## 运行时路径（代码自动设置）

| 组件 | 项目内路径 | 说明 |
|------|------------|------|
| InsightFace | `third_party/models/insightface/` | `FaceAnalysis(root=...)` |
| PaddleOCR | `third_party/models/paddleocr/` | 环境变量 `PADDLE_PDX_HOME` |
| YOLO 预训练 | `weights/pretrained/` | `yolov8n.pt` 等 |
| Ultralytics 配置 | `.ultralytics/` | 项目根目录 |

Python 包（`insightface`、`paddleocr`、`ultralytics`）仍通过 **pip 安装** 调用；`third_party/` 下的仓库用于 **阅读源码、移植、离线对照**。模型权重统一落在 `third_party/models/`。

## 与本项目模块对应

| 任务 | pip 包 | 源码 | 模型 |
|------|--------|------|------|
| 安全帽/车牌/动作 | ultralytics | yolov5 / ultralytics | weights/pretrained + weights/{task}/ |
| 人脸 | insightface | insightface/ | models/insightface/models/buffalo_l |
| 车牌 OCR | paddleocr | PaddleOCR/ | models/paddleocr/ |

## License

各上游仓库许可不同（AGPL / Apache 等），商用请分别确认。
