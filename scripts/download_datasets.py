#!/usr/bin/env python3
"""
下载开源数据集并整理到 datasets/ 目录。

数据源:
  - 人脸: LFW (Labeled Faces in the Wild)
  - 安全帽: 合成演示集 + SHWD 说明
  - 车牌: 合成演示集 + CCPD 说明
  - 动作: 合成演示集
"""

from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "datasets"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [跳过] {desc or dest.name} 已存在")
        return True
    print(f"  [下载] {desc or url}")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  [失败] {desc}: {e}")
        return False


def extract_archive(archive: Path, dest: Path) -> bool:
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as z:
                z.extractall(dest)
        elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".gz":
            with tarfile.open(archive, "r:*") as t:
                t.extractall(dest)
        return True
    except Exception as e:
        print(f"  [解压失败] {archive}: {e}")
        return False


def create_synthetic_face_dataset(persons: int = 5, images_per_person: int = 3) -> None:
    """网络不可达时生成合成人脸演示数据。"""
    print("\n=== 人脸合成演示数据集 ===")
    lfw_dir = DATASETS / "face" / "lfw"
    for p in range(persons):
        person_name = f"person_{p:02d}"
        person_dir = lfw_dir / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        base_color = np.random.randint(80, 200, 3).tolist()
        for i in range(images_per_person):
            img = np.random.randint(40, 120, (200, 200, 3), dtype=np.uint8)
            # 模拟面部椭圆区域
            cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, base_color, -1)
            cv2.circle(img, (75, 85), 8, (30, 30, 30), -1)
            cv2.circle(img, (125, 85), 8, (30, 30, 30), -1)
            cv2.ellipse(img, (100, 130), (20, 10), 0, 0, 180, (30, 30, 30), 2)
            cv2.imwrite(str(person_dir / f"{person_name}_{i:03d}.jpg"), img)
    print(f"  生成 {persons} 人 x {images_per_person} 张 -> {lfw_dir}")


def download_lfw() -> None:
    """下载 LFW 人脸数据集，失败则生成合成演示数据。"""
    print("\n=== 人脸数据集 (LFW) ===")
    lfw_dir = DATASETS / "face" / "lfw"
    lfw_dir.mkdir(parents=True, exist_ok=True)

    if any(lfw_dir.iterdir()):
        print(f"  [跳过] LFW 目录已有数据: {lfw_dir}")
        return

    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    archive = DATASETS / "face" / "lfw.tgz"
    ok = download_file(url, archive, "LFW")
    if ok and archive.exists():
        extract_archive(archive, DATASETS / "face")
        if not any(lfw_dir.iterdir()):
            for d in (DATASETS / "face").iterdir():
                if d.is_dir() and d.name not in ("gallery", "raw", "lfw"):
                    shutil.move(str(d), str(lfw_dir / d.name))
    if not any(lfw_dir.iterdir()):
        print("  LFW 下载失败，使用合成演示数据")
        create_synthetic_face_dataset()
    print(f"  LFW 目录: {lfw_dir}")


def _write_yolo_label(path: Path, class_id: int, cx: float, cy: float, w: float, h: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{class_id} {cx} {cy} {w} {h}\n")


def _create_synthetic_detection_dataset(
    task: str,
    classes: list[str],
    train_count: int = 40,
    val_count: int = 10,
) -> None:
    """生成合成 YOLO 检测数据集用于演示训练/推理流程。"""
    print(f"\n=== {task} 合成演示数据集 ===")
    base = DATASETS / task
    colors = {
        "helmet": [(0, 0, 255), (0, 255, 0)],
        "plate": [(255, 255, 0)],
        "action": [(128, 128, 128), (0, 165, 255), (0, 0, 255), (255, 0, 0)],
    }
    cls_colors = colors.get(task, [(255, 255, 255)] * len(classes))

    for split, count in [("train", train_count), ("val", val_count)]:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img = np.random.randint(60, 180, (480, 640, 3), dtype=np.uint8)
            cls_id = random.randint(0, len(classes) - 1)
            color = cls_colors[cls_id % len(cls_colors)]

            x1 = random.randint(80, 300)
            y1 = random.randint(60, 200)
            w = random.randint(80, 200)
            h = random.randint(80, 200)
            x2, y2 = min(x1 + w, 639), min(y1 + h, 479)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.putText(img, classes[cls_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cx = ((x1 + x2) / 2) / 640
            cy = ((y1 + y2) / 2) / 480
            bw = (x2 - x1) / 640
            bh = (y2 - y1) / 480

            fname = f"{task}_{split}_{i:04d}.jpg"
            cv2.imwrite(str(img_dir / fname), img)
            _write_yolo_label(lbl_dir / f"{task}_{split}_{i:04d}.txt", cls_id, cx, cy, bw, bh)

    data_yaml = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": {i: c for i, c in enumerate(classes)},
    }
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    print(f"  生成 {train_count} 训练 + {val_count} 验证样本 -> {base / 'data.yaml'}")


def create_plate_synthetic() -> None:
    _create_synthetic_detection_dataset("plate", ["plate"], 40, 10)


def create_helmet_synthetic() -> None:
    _create_synthetic_detection_dataset("helmet", ["no_helmet", "helmet"], 40, 10)


def create_action_synthetic() -> None:
    _create_synthetic_detection_dataset("action", ["normal", "smoking", "fighting", "falling"], 40, 10)


def write_dataset_readme() -> None:
    readme = DATASETS / "README.md"
    readme.write_text(
        """# 数据集目录说明

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
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="下载并准备数据集")
    parser.add_argument("--all", action="store_true", help="下载/生成全部数据集")
    parser.add_argument("--face", action="store_true")
    parser.add_argument("--helmet", action="store_true")
    parser.add_argument("--plate", action="store_true")
    parser.add_argument("--action", action="store_true")
    args = parser.parse_args()

    run_all = args.all or not any([args.face, args.helmet, args.plate, args.action])

    DATASETS.mkdir(parents=True, exist_ok=True)
    (DATASETS / "face" / "gallery").mkdir(parents=True, exist_ok=True)

    if run_all or args.face:
        download_lfw()
    if run_all or args.helmet:
        create_helmet_synthetic()
    if run_all or args.plate:
        create_plate_synthetic()
    if run_all or args.action:
        create_action_synthetic()

    write_dataset_readme()
    print("\n=== 数据集准备完成 ===")


if __name__ == "__main__":
    main()
