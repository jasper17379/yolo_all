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

| `--epochs` | 训练轮数，默认 20 |

| `--batch` | 批大小，默认 8 |

| `--resume-from-best` | 在 `weights/{task}/best_{yolo}{size}.pt` 上继续训练 |

| `--weights` | 显式指定 .pt 路径（覆盖 model-size 预训练） |

| `--resume` | 从 `last.pt` 断点续训 |



### 示例



```bash

# 预训练权重（nano）

python -m src.train.trainer --task plate --yolo yolov8 --model-size n --epochs 20



# 使用 small 预训练

python -m src.train.trainer --task plate --yolo yolov8 --model-size s --epochs 20



# 在 best_yolov8s 上继续训练

python -m src.train.trainer --task plate --yolo yolov8 --model-size s --epochs 50 --resume-from-best



# 指定预训练文件

python -m src.train.trainer --task helmet --yolo yolov10 --weights weights/pretrained/yolov10m.pt --epochs 20



# 安全帽 / 动作 / 人脸库

python -m src.train.trainer --task helmet --yolo yolov8 --model-size n --epochs 20

python -m src.train.trainer --task action --yolo yolov8 --model-size n --epochs 20

python -m src.train.trainer --task face

```



训练完成后：

- `runs/train/{task}_yolov8n/weights/best.pt`

- `weights/{task}/best_yolov8n.pt`（自动同步）



---



## 推理命令



```bash

python -m src.infer.inferencer --task helmet --source datasets/helmet/images/val --yolo yolov8 --model-size n

python -m src.infer.inferencer --task plate --source datasets/plate/images/val --yolo yolov8 --model-size n

python -m src.infer.inferencer --task action --source test_video.mp4 --yolo yolov8 --model-size s

python -m src.infer.inferencer --task face --source datasets/face/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg

python -m src.infer.inferencer --task helmet --source test.jpg --output-json outputs/result.json

```



---



## USB 摄像头实时 Demo



```bash

python demo.py

python demo.py --tasks helmet plate --yolo yolov8 --model-size n --no-face

python demo.py --lite

```



`--yolo` 与 `--model-size` 决定加载的 `weights/{task}/best_{tag}.pt`。



---



## REST API



```bash

python -m src.api.server

```



训练/推理 JSON 支持 `model_size` 字段（默认 `"n"`），与命令行 `--model-size` 一致。



---



## 模型导出



```bash

python scripts/export_models.py --task plate --yolo yolov8 --model-size n --format onnx

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

