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

## 开源数据集来源

| 任务 | 数据集 | 链接 |
|------|--------|------|
| 人脸识别 | LFW | http://vis-www.cs.umass.edu/lfw/ |
| 安全帽 | SHWD | https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset |
| 车牌 | CCPD | https://github.com/detectRecog/CCPD |
| 动作 | UCF-Crime | https://www.crcv.ucf.edu/projects/real-world/ |

## 添加自定义数据

1. **检测任务**: 将图片放入 `custom/images/{task}/`，YOLO 标签放入 `custom/labels/{task}/`
2. **人脸录入**: 使用 API `POST /api/v1/face/enroll` 或命令行
3. 合并自定义数据后重新运行训练

## 下载命令

```bash
python scripts/download_datasets.py --all
```
