# third_party — 业务依赖源码与模型

本目录集中存放 **检测框架 + 业务场景** 的源码与模型权重，避免依赖散落在系统目录或用户家目录。

## 目录结构

```
third_party/
├── yolov5/                    # YOLOv5 源码（参考/C++ 对照）
├── ultralytics/               # YOLOv8/v10 运行库（vendored）
├── insightface/               # InsightFace 源码（vendored）
│   └── python-package/insightface/
├── HyperLPR/                  # HyperLPR3 源码（vendored）
│   └── Prj-Python/hyperlpr3/
├── PaddleOCR/                 # PaddleOCR 源码（参考）
├── labelImg-master/           # 标注工具
└── models/
    ├── insightface/models/buffalo_l/
    ├── hyperlpr3/
    └── paddleocr/
```

## 无需 pip install 的包

| 包 | 源码路径 | 加载方式 |
|----|----------|----------|
| ultralytics | `third_party/ultralytics/` | `bootstrap_env()` → sys.path |
| hyperlpr3 | `third_party/HyperLPR/Prj-Python/` | sys.path + 模型 `models/hyperlpr3/` |
| insightface | `third_party/insightface/python-package/` | sys.path + 模型 `models/insightface/` |

## 一键初始化

```bash
python scripts/setup_third_party.py
python scripts/setup_yolo_sources.py
python scripts/setup_third_party.py --download-hyperlpr-models --download-face-model
python scripts/check_offline_deps.py
```

## C++ 部署

边缘设备 C++ 推理不需要上述 Python 包，直接使用：

- YOLO：`.onnx` / `.rknn`（见 `deploy/`）
- 人脸：加载 `models/insightface/models/buffalo_l/*.onnx`（ONNX Runtime）
- 车牌：HyperLPR C++ 或 HyperLPR3 ONNX

详见 [deploy/README.md](../deploy/README.md)。

## 运行时环境变量

| 组件 | 变量 | 目录 |
|------|------|------|
| InsightFace | `INSIGHTFACE_HOME` | `models/insightface/` |
| HyperLPR3 | `HYPERLPR3_HOME` | `models/hyperlpr3/` |
| PaddleOCR | `PADDLE_PDX_CACHE_HOME` | `models/paddleocr/` |
| Ultralytics | `YOLO_CONFIG_DIR` | `ultralytics_config/` |

## License

各上游仓库许可不同，商用请分别确认。
