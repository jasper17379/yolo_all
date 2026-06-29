# Vision AI Platform

集成 **YOLOv5 / YOLOv8 / YOLOv10** 可选框架，并提供 **人脸识别、车牌识别、安全帽检测、人体动作识别**（抽烟、打人、跌倒等）的统一训练与推理平台。

## 功能概览

| 模块 | 技术方案 | 类型 |
|------|----------|------|
| 目标检测框架 | YOLOv5 / v8 / v10 (Ultralytics) | 可切换 |
| 安全帽检测 | YOLO 检测 | `no_helmet`, `helmet` |
| 车牌识别 | YOLO 检测 + PaddleOCR | 检测 + OCR |
| 动作识别 | YOLO 检测 | `normal`, `smoking`, `fighting`, `falling` |
| 人脸识别 | InsightFace (buffalo_l) | 检测 + 特征比对 |

## 项目结构

```
yolo8_10_all/
├── configs/                 # 全局与任务配置
│   ├── global.yaml
│   └── tasks/               # helmet, plate, action, face
├── datasets/                # 数据集统一目录
│   ├── face/lfw/            # LFW 开源人脸数据
│   ├── face/gallery/        # 人脸库(录入)
│   ├── helmet/              # 安全帽 YOLO 数据
│   ├── plate/               # 车牌 YOLO 数据
│   ├── action/              # 动作 YOLO 数据
│   └── custom/              # 自定义追加数据
├── src/
│   ├── core/                # YOLO 封装、配置
│   ├── train/               # 训练入口
│   ├── infer/               # 推理入口
│   ├── tasks/               # 人脸/车牌业务模块
│   └── api/                 # REST API
├── scripts/                 # 数据集下载、验证、导出
├── deploy/
│   ├── closed_source/       # 闭源部署文档
│   ├── cpp/                 # C++ ONNX 推理 (Ubuntu/RK3588)
│   └── Dockerfile
├── weights/                 # 训练产出权重
├── runs/                    # 训练/推理运行记录
└── outputs/                 # 验证报告与 JSON 输出
```

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

> **注意**: PaddleOCR 首次运行会自动下载 OCR 模型；InsightFace 首次运行会下载 `buffalo_l` 模型。

### 2. 下载/准备数据集

```bash
# 方式 A: 从外部真实数据导入 (推荐)
python scripts/import_external_datasets.py --helmet-src E:\iVS-100-DB-Bak\datasets --face-src E:\iVS-100-DB-Bak\face\face\test

# 仅重新导入人脸
python scripts/import_external_datasets.py --skip-helmet --face-src E:\iVS-100-DB-Bak\face\face\test

python scripts/import_external_datasets.py --skip-helmet --face-src E:\iVS-100-DB-Bak\face\face\train
# 方式 B: 自动下载/生成演示数据 (网络不可达时生成合成方块图，仅用于流程验证)
python scripts/download_datasets.py --all
```

| 任务 | 数据来源 |
|------|----------|
| 人脸 | [LFW](http://vis-www.cs.umass.edu/lfw/) |
| 安全帽 | 演示集 + [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) |
| 车牌 | 演示集 + [CCPD](https://github.com/detectRecog/CCPD) |
| 动作 | 演示集 + [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) |

详细说明见 [datasets/README.md](datasets/README.md)。

---

## 训练命令

### 通用参数

| 参数 | 说明 |
|------|------|
| `--task` | `helmet` / `plate` / `action` / `face` / `all` |
| `--yolo` | `yolov5` / `yolov8` / `yolov10` |
| `--epochs` | 训练轮数，默认 20 |
| `--batch` | 批大小，默认 8 |
| `--resume-from-best` | 在上次训练的 `best.pt` 基础上加强训练 |
| `--weights` | 指定原始预训练权重 (如 `yolov8n.pt` 或自定义路径) |
| `--resume` | 从 `last.pt` 断点续训 |

### 各任务训练示例

```bash
# 安全帽检测 - YOLOv8, 20轮, 从预训练权重开始
python -m src.train.trainer --task helmet --yolo yolov8 --epochs 20

# 安全帽 - 在上次 best 权重基础上继续训练
python -m src.train.trainer --task helmet --yolo yolov8 --epochs 50 --resume-from-best

# 安全帽 - 指定自定义预训练权重
python -m src.train.trainer --task helmet --yolo yolov10 --weights yolov10n.pt --epochs 20

# 车牌检测
python -m src.train.trainer --task plate --yolo yolov8 --epochs 20

# 动作识别 (抽烟/打人/跌倒)
python -m src.train.trainer --task action --yolo yolov8 --epochs 20

# 人脸库构建 (从 LFW 初始化 gallery)
python -m src.train.trainer --task face
# 识别人脸
python -m src.infer.inferencer --task face --source E:\iVS-100-DB-Bak\face\test\geyou_15.jpg
python -m src.infer.inferencer --task face --source E:\iVS-100-DB-Bak\face\test\pengyuyan_11.jpg

###{
      # 仅重新导入人脸liu 
      python scripts/import_external_datasets.py --skip-helmet --face-src E:\iVS-100-DB-Bak\face\face\train
      # 重建人脸库
      python -m src.train.trainer --task face
      # 识别人脸
      python -m src.infer.inferencer --task face --source datasets/face/lfw/jiangwen/jiangwen_10.jpg
}
# 一次训练全部检测任务
python -m src.train.trainer --task all --yolo yolov8 --epochs 20
```

训练完成后权重保存在:
- `runs/train/{task}/weights/best.pt`
- `weights/{task}/best.pt` (自动同步)

---

## 推理命令

```bash
# 安全帽检测
python -m src.infer.inferencer --task helmet --source datasets/helmet/images/val --yolo yolov8

# 车牌识别 (检测 + OCR)
python -m src.infer.inferencer --task plate --source datasets/plate/images/val --yolo yolov8

# 动作识别
python -m src.infer.inferencer --task action --source test_video.mp4 --yolo yolov8

# 人脸识别
python -m src.infer.inferencer --task face --source datasets/face/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg

# 输出 JSON 结果
python -m src.infer.inferencer --task helmet --source test.jpg --output-json outputs/result.json

# 切换 YOLO 版本
python -m src.infer.inferencer --task helmet --source test.jpg --yolo yolov10
```

推理可视化结果保存在 `runs/predict/{task}/`。

---

## 扩展接口

### REST API 服务

```bash
# 启动 API 服务
python -m src.api.server
# 默认 http://0.0.0.0:8000
```

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/train` | POST | 触发训练 |
| `/api/v1/infer` | POST | 推理 |
| `/api/v1/face/enroll` | POST | 录入新人脸 (multipart: name + image) |
| `/api/v1/face/list` | GET | 列出已录入人脸 |
| `/api/v1/face/{name}` | DELETE | 删除人脸 |
| `/api/v1/data/add` | POST | 添加自定义训练数据 |

### 人脸录入示例

```bash
curl -X POST http://localhost:8000/api/v1/face/enroll \
  -F "name=张三" \
  -F "image=@photo.jpg"
```

### 添加训练数据示例

```bash
curl -X POST http://localhost:8000/api/v1/data/add \
  -F "task=helmet" \
  -F "class_id=1" \
  -F "image=@new_sample.jpg"
```

自定义数据目录: `datasets/custom/images/{task}/` 与 `datasets/custom/labels/{task}/`

---

## USB 摄像头实时 Demo

```bash
# 多模型实时推理 (安全帽+车牌+动作+人脸)，红框标注，按 Q 退出
python demo.py

# 仅检测部分任务 (更快)
python demo.py --tasks helmet plate action --no-face

# 指定摄像头
python demo.py --camera 0 --conf 0.35 --imgsz 416
```

## 真实车牌/动作数据

```bash
# 从图库下载真实图片并自动标注 (Wikimedia/Pexels，失败时用本地工地图补充)
python scripts/download_real_plate_action.py

# 重新训练
python -m src.train.trainer --task plate --epochs 20
python -m src.train.trainer --task action --epochs 20
```

```bash
# 自动: 下载数据 -> 各任务训练20轮 -> 测试集推理 -> 生成报告
python scripts/verify_pipeline.py
```

报告输出: `outputs/verify_report.json`

---

## 模型导出与部署

### 导出 ONNX

```bash
python scripts/export_models.py --task all --format onnx
```

### 部署文档

| 场景 | 文档 |
|------|------|
| 闭源 / Docker 生产部署 | [deploy/closed_source/README.md](deploy/closed_source/README.md) |
| C++ ONNX 推理 (Ubuntu + NVIDIA) | [deploy/cpp/README.md](deploy/cpp/README.md) |
| RK3588 NPU 部署 | [deploy/cpp/README.md](deploy/cpp/README.md) |

### Docker 快速部署

```bash
docker build -t vision-ai:1.0 -f deploy/Dockerfile .
docker run -d -p 8000:8000 --gpus all vision-ai:1.0
```

---

## YOLO 版本选择建议

| 版本 | 特点 | 适用场景 |
|------|------|----------|
| YOLOv5 | 成熟稳定、社区资源多 | 兼容性优先 |
| YOLOv8 | 精度与速度均衡 (默认) | 通用推荐 |
| YOLOv10 | 无 NMS 推理更快 | 实时性要求高 |

切换方式: 训练/推理时加 `--yolo yolov5|yolov8|yolov10`

---

## 常见问题

**Q: 如何使用真实大规模数据集?**  
A: 参考 `datasets/README.md` 下载 SHWD/CCPD 等，转换为 YOLO 格式后放入对应目录，更新 `data.yaml`。

**Q: 如何在上次训练基础上加强?**  
A: 使用 `--resume-from-best`，将从 `weights/{task}/best.pt` 或最新 `runs/train/{task}/weights/best.pt` 继续。

**Q: 如何完全重新训练?**  
A: 不加 `--resume-from-best`，使用 `--weights yolov8n.pt` 指定预训练权重。

---

## License

本项目代码 MIT 许可。第三方模型与数据集请遵循各自许可协议。



