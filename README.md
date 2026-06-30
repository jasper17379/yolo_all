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

yolo_all/

├── configs/                 # 全局与任务配置

├── datasets/                # 数据集（见 datasets/README.md）

├── src/

│   ├── core/                # YOLO 封装、配置、权重路径 (weights.py)

│   ├── train/               # 训练入口

│   ├── infer/               # 推理入口

│   ├── tasks/               # 人脸/车牌业务模块

│   └── api/                 # REST API

├── third_party/             # YOLO 官方源码（移植/C++ 对照，见 third_party/README.md）
│   ├── yolov5/              # YOLOv5 独立仓库
│   └── ultralytics/         # YOLOv8 + YOLOv10
├── scripts/
│   ├── setup_yolo_sources.py           # 拉取/更新 third_party 源码
│   ├── build_reference_datasets.py     # 重建车牌/动作参考范例（清理错误旧数据）
│   ├── download_pretrained_weights.py
│   └── import_plate_dataset.py

├── weights/

│   ├── pretrained/          # yolov8n.pt, yolov8s.pt ... 官方预训练

│   └── {task}/              # best_yolov8n.pt 等训练产出

├── runs/train/{task}_yolov8n/  # 按任务+版本+规格命名的训练 run

└── demo.py

```



## 快速开始



### 1. 环境安装



```bash

python -m venv venv

venv\Scripts\activate          # Windows

pip install -r requirements.txt

```



### 2.1 获取 YOLO 官方源码（移植 / C++ 对照）

本项目训练通过 `pip` 调用各库，**源码与模型权重** 统一放在 `third_party/`：

```bash
python scripts/setup_third_party.py    # InsightFace/PaddleOCR 源码 + 人脸模型迁移
python scripts/setup_yolo_sources.py     # YOLOv5 / ultralytics 源码
```

| 路径 | 内容 |
|------|------|
| `third_party/models/insightface/` | 人脸模型 buffalo_l（不再用 `~/.insightface`） |
| `third_party/models/paddleocr/` | 车牌 OCR 模型缓存 |
| `third_party/yolov5/`、`ultralytics/` | YOLO 源码 |
| `third_party/insightface/`、`PaddleOCR/` | 业务库源码 |

详见 [third_party/README.md](third_party/README.md)。

### 2. 下载 YOLO 预训练权重（推荐）



将 v5/v8/v10 常用规格下载到 `weights/pretrained/`，训练时可直接 `--model-size s` 而无需重复下载：



```bash

# 下载全部常用规格 (n/s/m/l/x，v10 含 b)

python scripts/download_pretrained_weights.py



# 仅下载指定权重

python scripts/download_pretrained_weights.py --names yolov8n.pt yolov8s.pt yolov10m.pt

```



目录示例：



```

weights/pretrained/

  yolov5n.pt  yolov5s.pt  ...

  yolov8n.pt  yolov8s.pt  ...

  yolov10n.pt yolov10b.pt ...

```



### 3. 下载/准备数据集



```bash

# 车牌/动作：重建参考范例（会清理旧错误数据，见下文）
python scripts/build_reference_datasets.py

# 安全帽 + 人脸（外部路径）
python scripts/import_external_datasets.py --helmet-src E:\iVS-100-DB-Bak\datasets --face-src E:\iVS-100-DB-Bak\face\face\test

# 车牌外部数据格式检测
python scripts/import_plate_dataset.py --src E:\iVS-100-DB-Bak\datasets\plate --analyze-only

# 安全帽等其它任务演示数据
python scripts/download_datasets.py --all
```

---

## 车牌 / 动作参考范例数据（LabelImg 模板）

旧的 `download_real_plate_action.py` 自动框标注不准确，**已弃用**。请使用：

```bash
python scripts/build_reference_datasets.py
```

会清理 `datasets/plate`、`datasets/action` 下错误图片，并生成：

```
datasets/plate/reference/
  images/          # 范例图
  labels/          # 对应 YOLO txt（框与图严格对齐）
  preview/         # 红框预览图，用于核对
  classes.txt      # LabelImg 类别：plate
  README.txt       # 标注说明

datasets/action/reference/
  classes.txt      # normal / smoking / fighting / falling
```

**自采数据流程：**

1. 用 LabelImg 打开 `reference/images`，格式选 YOLO，加载 `classes.txt`
2. 对照 `preview/` 红框理解框选范围
3. 新图放入 `datasets/plate/images/train` + `labels/train`（动作同理）
4. 重新训练

当前范例为本地生成的精确标注图（网络图下载受限时仍可训练）。数据量很少，仅作格式模板；量产请自行标注 50+ 张。

```bash
# 范例数据训练验证（10 epoch）
python -m src.train.trainer --task plate --yolo yolov8 --model-size n --epochs 10 --batch 4
python -m src.train.trainer --task action --yolo yolov8 --model-size n --epochs 10 --batch 4
python -m src.infer.inferencer --task plate --source datasets/plate/images/val --yolo yolov8 --model-size n --conf 0.01
python -m src.infer.inferencer --task action --source datasets/action/images/val --yolo yolov8 --model-size n --conf 0.01
```

---

## 权重命名与 `--model-size`



训练、推理、Demo 通过 **`--yolo` + `--model-size`** 对齐同一套权重：



| 参数 | 示例 | 对应预训练 | 训练产出 best |

|------|------|------------|---------------|

| `--yolo yolov8 --model-size n` | 默认 | `weights/pretrained/yolov8n.pt` | `weights/plate/best_yolov8n.pt` |

| `--yolo yolov8 --model-size s` | 小模型 | `weights/pretrained/yolov8s.pt` | `weights/plate/best_yolov8s.pt` |

| `--yolo yolov10 --model-size m` | 中模型 | `weights/pretrained/yolov10m.pt` | `weights/helmet/best_yolov10m.pt` |



训练 run 目录：`runs/train/plate_yolov8s/`（不再与不同规格混在同一个 `plate/` 目录）。



**继续训练**（在已有 best 上加强，须与版本+规格一致）：



```bash

python -m src.train.trainer --task plate --yolo yolov8 --model-size s --epochs 50 --resume-from-best

```



**推理**（须与训练时 `--yolo`、`--model-size` 一致）：



```bash

python -m src.infer.inferencer --task plate --source datasets/plate/images/val --yolo yolov8 --model-size s

```



旧版 `weights/{task}/best.pt` 仍可作为回退，但新训练请使用 `best_{yolo}{size}.pt`。



---



## 车牌数据集说明



`E:\iVS-100-DB-Bak\datasets\plate` 若为 **按字母/汉字分子文件夹、20×20 单字图**，属于 **字符分类/OCR 数据**，**不能**用于 YOLO 车牌定位训练。



| 数据类型 | 结构 | 能否训练 YOLO 定位 |

|----------|------|-------------------|

| 字符分类 | `plate/A/*.jpg` 20×20 单字 | 否 → 用于 OCR，非本项目 YOLO 阶段 |

| YOLO 检测 | `images/` + `labels/` 场景图+框 | 是 |



本项目流程：**YOLO 框车牌** → **PaddleOCR 读字**。



YOLO 检测数据格式与标注工具见 [datasets/README.md](datasets/README.md)。



导入 YOLO 格式车牌数据：



```bash

python scripts/import_plate_dataset.py --src <YOLO格式目录>

python -m src.train.trainer --task plate --yolo yolov8 --model-size n --epochs 20

```



推荐真实数据：[CCPD](https://github.com/detectRecog/CCPD) 转 YOLO，或 LabelImg 在场景图上标注 `plate`。



---



## 训练命令

### 通用参数

| 参数 | 说明 |
|------|------|
| `--task` | `helmet` / `plate` / `action` / `face` / `all` |
| `--yolo` | `yolov5` / `yolov8` / `yolov10` |
| `--model-size` | `n` / `s` / `m` / `l` / `x`（v10 另有 `b`） |
| `--device` | `auto` / `cpu` / `cuda` / `cuda:0` / `0` / `0,1`（多卡） |
| `--epochs` | 训练轮数，默认 20 |
| `--batch` | 批大小，默认 8 |
| `--imgsz` | 输入尺寸，默认 640 |
| `--resume-from-best` | 在 `weights/{task}/best_{yolo}{size}.pt` 上继续训练 |
| `--weights` | 显式指定 .pt 路径（覆盖 model-size 预训练） |
| `--resume` | 从 `last.pt` 断点续训 |

### 训练超参（可选，默认见 `configs/global.yaml`）

| 参数 | 说明 | 默认 |
|------|------|------|
| `--lr0` | 初始学习率 | 0.01 |
| `--lrf` | 最终学习率因子 | 0.01 |
| `--patience` | 早停耐心值（epoch） | 50 |
| `--workers` | DataLoader 线程数 | 4 |
| `--optimizer` | `auto` / `SGD` / `Adam` / `AdamW` | auto |
| `--momentum` | SGD 动量 | 0.937 |
| `--weight-decay` | 权重衰减 | 0.0005 |
| `--warmup-epochs` | 预热轮数 | 3.0 |
| `--mosaic` | 马赛克增强概率 0~1 | 1.0 |
| `--close-mosaic` | 最后 N 个 epoch 关闭 mosaic | 10 |
| `--cos-lr` | 余弦学习率调度 | 关 |
| `--no-amp` | 关闭混合精度 AMP | AMP 开 |
| `--freeze` | 冻结 backbone 前 N 层 | 不冻结 |
| `--seed` | 随机种子 | 0 |

全局默认可在 `configs/global.yaml` 的 `device`、`train`、`infer` 段修改。

### 设备选择

```bash
# 查看当前环境可用设备
python scripts/list_devices.py

# 自动：有 GPU 用 cuda:0，否则 cpu
python -m src.train.trainer --task helmet --device auto

# 指定单卡 / 多卡 / 强制 CPU
python -m src.train.trainer --task helmet --device 0
python -m src.train.trainer --task helmet --device 0,1 --batch 16
python -m src.train.trainer --task face --device cpu
```

| `--device` | YOLO 训练/推理 | InsightFace 人脸 | PaddleOCR 车牌 |
|------------|----------------|------------------|----------------|
| `auto` | 有 GPU → `0`，否则 `cpu` | CUDA 优先，失败回退 CPU | 尝试 GPU |
| `cpu` | 强制 CPU | 仅 CPU | CPU |
| `0` / `cuda:0` | 指定 GPU 0 | GPU 0 | GPU |
| `0,1` | 多卡 DataParallel | 使用 GPU 0 | GPU |

### 示例

```bash
# 预训练权重（nano）+ 自定义超参
python -m src.train.trainer --task plate --yolo yolov8 --model-size n \
  --device 0 --epochs 20 --batch 8 --lr0 0.01 --patience 30

# 使用 small 预训练
python -m src.train.trainer --task plate --yolo yolov8 --model-size s --device auto --epochs 20

# 在 best_yolov8s 上继续训练
python -m src.train.trainer --task plate --yolo yolov8 --model-size s --device 0 \
  --epochs 50 --resume-from-best --lr0 0.001

# 指定预训练文件
python -m src.train.trainer --task helmet --yolo yolov10 --weights weights/pretrained/yolov10m.pt \
  --device 0 --epochs 20 --optimizer AdamW

# 安全帽 / 动作 / 人脸库
python -m src.train.trainer --task helmet --yolo yolov8 --model-size n --device auto --epochs 20
python -m src.train.trainer --task action --yolo yolov8 --model-size n --device 0 --epochs 20 --no-amp
python -m src.train.trainer --task face --device cuda:0
```



训练完成后：

- `runs/train/{task}_yolov8n/weights/best.pt`

- `weights/{task}/best_yolov8n.pt`（自动同步）



---



## 推理命令

| 参数 | 说明 | 默认 |
|------|------|------|
| `--device` | 同训练：`auto` / `cpu` / `0` / `cuda:0` | `auto` |
| `--conf` | 置信度阈值 | 0.25 |
| `--iou` | NMS IoU 阈值 | 0.45 |
| `--imgsz` | 输入尺寸 | 640 |
| `--half` | FP16 推理（需 GPU） | 关 |

```bash
python -m src.infer.inferencer --task helmet --source datasets/helmet/images/val \
  --yolo yolov8 --model-size n --device auto

python -m src.infer.inferencer --task plate --source datasets/plate/images/val \
  --device 0 --conf 0.3 --imgsz 640

python -m src.infer.inferencer --task action --source test_video.mp4 \
  --yolo yolov8 --model-size s --device 0 --half

python -m src.infer.inferencer --task face --source datasets/face/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg \
  --device cpu

python -m src.infer.inferencer --task helmet --source test.jpg --device 0 --output-json outputs/result.json
```



---



## USB 摄像头实时 Demo



```bash
python demo.py
python demo.py --tasks helmet plate --yolo yolov8 --model-size n --device 0 --no-face
python demo.py --lite --device cpu
```

`--yolo`、`--model-size` 决定加载的 `weights/{task}/best_{tag}.pt`；`--device` 同时作用于 YOLO 检测与人脸 InsightFace。



---



## REST API



```bash

python -m src.api.server

```



训练/推理 JSON 除 `model_size` 外，还支持 `device` 及主要超参：

```json
POST /api/v1/train
{
  "task": "helmet",
  "yolo_version": "yolov8",
  "model_size": "n",
  "device": "0",
  "epochs": 20,
  "batch": 8,
  "imgsz": 640,
  "lr0": 0.01,
  "patience": 30
}

POST /api/v1/infer
{
  "task": "helmet",
  "source": "datasets/helmet/images/val",
  "device": "auto",
  "conf": 0.25,
  "iou": 0.45,
  "half": false
}
```

`GET /health` 返回 `gpu_count` 与 `default_device`。



---



## 模型导出



```bash
python scripts/export_models.py --task plate --yolo yolov8 --model-size n --format onnx --device 0
```



---



## YOLO 版本与规格



| 版本 | 常用规格 | 特点 |

|------|----------|------|

| YOLOv5 | n/s/m/l/x | 成熟稳定 |

| YOLOv8 | n/s/m/l/x | 默认推荐 |

| YOLOv10 | n/s/m/b/l/x | 推理更快 |



规格越大精度一般越高、速度越慢。同一任务可同时保留 `best_yolov8n.pt` 与 `best_yolov8s.pt`，按需选用。



---



## 常见问题



**Q: 外部 plate 按字母分文件夹的数据能训练吗？**  

A: 不能用于 YOLO 定位。那是单字分类数据；请使用 CCPD 或 LabelImg 标注的场景图+车牌框。运行 `python scripts/import_plate_dataset.py --analyze-only` 可自动检测。



**Q: 如何换不同规模的模型训练？**  

A: 先 `python scripts/download_pretrained_weights.py`，再 `--model-size s/m/l`。各规格 best 分开保存，互不覆盖。



**Q: 推理结果不对？**  

A: 检查 `--yolo` 和 `--model-size` 是否与训练时一致。



---



## License



本项目代码 MIT 许可。第三方模型与数据集请遵循各自许可协议。

