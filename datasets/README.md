# 数据集目录说明



## 目录结构



```

datasets/

├── face/

│   ├── lfw/              # LFW 开源人脸数据集

│   ├── gallery/          # 人脸库(录入的人脸特征)

│   └── raw/              # 原始人脸图片(可扩展)

├── helmet/               # 安全帽检测 (YOLO)

├── plate/                # 车牌检测 (YOLO)

├── action/               # 动作识别 (YOLO)

└── custom/               # 自定义追加训练数据

    ├── images/{task}/

    └── labels/{task}/

```



## 车牌数据（检测 + 识别）



本项目 **plate 任务分两阶段**：

1. **检测（YOLO，可训练）**：在场景图中框出车牌区域
2. **识别（HyperLPR3，预训练）**：读取车牌号，无需训练



### 检测标注（YOLO）



```

datasets/plate/

  images/train/*.jpg

  labels/train/*.txt    # 与图片同名，YOLO 格式

  images/val/

  labels/val/

  recognition/train/*.json   # 可选：车牌号真值（OCR 评测用）

  recognition/val/*.json

  data.yaml

```



标签 `labels/train/xxx.txt` 每行（归一化 0~1）：



```

class_id x_center y_center width height

```



车牌检测通常 `class_id=0`，类别名 `plate`。



### 识别标注（JSON，与检测分开）



`recognition/train/xxx.json` 示例：



```json

{

  "version": "1.0",

  "image": "scene_001.jpg",

  "plates": [

    {

      "plate_text": "京A12345",

      "plate_type": "蓝牌",

      "bbox_yolo": [0, 0.5, 0.8, 0.3, 0.09],

      "note": "与 labels 中检测框一一对应"

    }

  ]

}

```



模板见 `datasets/plate/reference/templates/`。识别标注用于 **OCR 准确率评测**，不用于训练。



```bash

python scripts/eval_plate_recognition.py --split val --mode crop

```



### 不能直接使用的情况



若数据为 **按字符/汉字分子文件夹**（如 `plate/A/*.jpg`、`plate/京/*.jpg`），且图片为 **20×20 左右单字裁剪图**，则属于 **OCR/字符分类数据**，不能用于 YOLO 车牌定位训练。本项目识别阶段使用预训练 **HyperLPR3**（回退 PaddleOCR），无需此类单字数据训练。



检测命令：



```bash

python scripts/import_plate_dataset.py --src E:\iVS-100-DB-Bak\datasets\plate --analyze-only

```



### 推荐数据来源



| 来源 | 说明 |

|------|------|

| [CCPD](https://github.com/detectRecog/CCPD) | 中国城市车牌，含矩形/四点标注，需转为 YOLO |

| 自建 | 用 LabelImg 等工具在场景图上框车牌 |



### 推荐标注工具



| 工具 | 说明 |

|------|------|

| [LabelImg](https://github.com/HumanSignal/labelImg) | 桌面工具，导出 YOLO txt |

| [Roboflow](https://roboflow.com/) | 在线标注，导出 YOLO |

| [CVAT](https://www.cvat.ai/) | 团队协作标注 |

| [makesense.ai](https://www.makesense.ai/) | 浏览器免费标注 |



导入 YOLO 格式数据：



```bash

python scripts/import_plate_dataset.py --src <含images和labels的目录>

```



## 开源数据集来源



| 任务 | 数据集 | 链接 |

|------|--------|------|

| 人脸识别 | LFW | http://vis-www.cs.umass.edu/lfw/ |

| 安全帽 | SHWD | https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset |

| 车牌 | CCPD | https://github.com/detectRecog/CCPD |

| 动作 | UCF-Crime | https://www.crcv.ucf.edu/projects/real-world/ |



## 添加自定义数据



1. **检测任务**: 参照 `datasets/plate/reference/` 或 `datasets/action/reference/` 用 LabelImg 标注，再放入 `images/train` + `labels/train`

2. **人脸录入**: 使用 API `POST /api/v1/face/enroll` 或命令行

3. 合并自定义数据后重新运行训练

重建范例: `python scripts/build_reference_datasets.py`



## 下载命令



```bash
python scripts/download_datasets.py --all
python scripts/build_reference_datasets.py   # 车牌/动作参考范例
```

