#!/usr/bin/env python3
"""
检测并导入车牌 YOLO 检测数据集。

支持:
  - YOLO 格式: images/ + labels/（与 helmet 相同）
  - CCPD 转换后的 YOLO 目录

不支持（会给出说明）:
  - 按字符分子文件夹的分类数据集（如 plate/A/*.jpg 20x20 单字图）
    该类数据用于 OCR/字符识别，不能训练 YOLO 车牌定位。
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = PROJECT_ROOT / "datasets"
DEFAULT_PLATE_SRC = Path(r"E:\iVS-100-DB-Bak\datasets\plate")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def detect_dataset_type(src: Path) -> tuple[str, dict]:
    """返回 (类型, 详情字典)。"""
    info: dict = {"path": str(src)}

    src_img = src / "images"
    src_lbl = src / "labels"
    if src_img.is_dir() and src_lbl.is_dir():
        pairs = [
            p.stem
            for p in src_img.iterdir()
            if p.suffix.lower() in IMG_EXT and (src_lbl / f"{p.stem}.txt").exists()
        ]
        if pairs:
            info["pairs"] = len(pairs)
            return "yolo", info

    subdirs = [d for d in src.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not subdirs:
        return "unknown", info

    sample_sizes: set[tuple[int, int]] = set()
    img_count = 0
    for d in subdirs[:8]:
        for p in list(d.glob("*.jpg"))[:5] + list(d.glob("*.png"))[:2]:
            try:
                import cv2

                im = cv2.imread(str(p))
                if im is not None:
                    h, w = im.shape[:2]
                    sample_sizes.add((w, h))
                    img_count += 1
            except Exception:
                pass

    for d in subdirs:
        img_count += len([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT])

    info["folders"] = len(subdirs)
    info["sample_folder_names"] = [d.name for d in subdirs[:12]]
    info["image_count"] = img_count
    info["sample_sizes"] = list(sample_sizes)

    if subdirs and sample_sizes and max(max(s) for s in sample_sizes) <= 64:
        return "char_classification", info

    if subdirs and img_count > 0:
        return "char_classification", info

    return "unknown", info


def _copy_yolo_pairs(stems: list[str], src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path, prefix: str) -> int:
    count = 0
    for i, stem in enumerate(stems):
        img_src = src_img / f"{stem}.jpg"
        if not img_src.exists():
            for ext in (".jpeg", ".png", ".JPG", ".JPEG", ".bmp"):
                alt = src_img / f"{stem}{ext}"
                if alt.exists():
                    img_src = alt
                    break
        lbl_src = src_lbl / f"{stem}.txt"
        if not img_src.exists() or not lbl_src.exists():
            continue
        name = f"{prefix}_{i:04d}{img_src.suffix.lower()}"
        shutil.copy2(img_src, dst_img / name)
        shutil.copy2(lbl_src, dst_lbl / f"{Path(name).stem}.txt")
        count += 1
    return count


def import_plate_yolo(src: Path, val_ratio: float = 0.2, seed: int = 42) -> dict:
    """导入 YOLO 格式车牌检测数据（class 0 = plate）。"""
    src_img = src / "images"
    src_lbl = src / "labels"
    if not src_img.is_dir() or not src_lbl.is_dir():
        raise FileNotFoundError(f"需要 YOLO 目录结构: {src}/images 与 {src}/labels")

    stems = sorted(
        p.stem
        for p in src_img.iterdir()
        if p.suffix.lower() in IMG_EXT and (src_lbl / f"{p.stem}.txt").exists()
    )
    if not stems:
        raise FileNotFoundError(f"未找到匹配的图像/标签: {src}")

    random.seed(seed)
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * val_ratio))
    val_stems = stems[:n_val]
    train_stems = stems[n_val:]

    base = DATASETS / "plate"
    for split in ("train", "val"):
        _clear_dir(base / "images" / split)
        _clear_dir(base / "labels" / split)

    n_train = _copy_yolo_pairs(train_stems, src_img, src_lbl, base / "images" / "train", base / "labels" / "train", "plate_train")
    n_val = _copy_yolo_pairs(val_stems, src_img, src_lbl, base / "images" / "val", base / "labels" / "val", "plate_val")

    data_yaml = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "plate"},
    }
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)

    return {"train": n_train, "val": n_val, "total": len(stems)}


def print_char_classification_help(info: dict) -> None:
    print("\n" + "=" * 60)
    print("当前数据集类型: 字符分类（按文件夹名标定单字/字符）")
    print("=" * 60)
    print(f"路径: {info.get('path')}")
    print(f"子文件夹数: {info.get('folders')}, 图片约: {info.get('image_count')}")
    print(f"示例子文件夹: {info.get('sample_folder_names')}")
    print(f"抽样尺寸: {info.get('sample_sizes')} (多为 20x20 单字裁剪图)")
    print()
    print("结论: 不能直接用于本项目的 YOLO 车牌「定位」训练。")
    print("  - 本项目 plate 任务: YOLO 在完整场景中框出车牌区域 → PaddleOCR 读字")
    print("  - 您的数据: 已是单字符小图 + 文件夹类别名，属于 OCR/字符分类数据")
    print()
    print("YOLO 车牌检测所需格式:")
    print("  datasets/plate/")
    print("    images/train/*.jpg    # 含车牌的完整场景图")
    print("    labels/train/*.txt    # 同名 YOLO 标签")
    print("    images/val/")
    print("    labels/val/")
    print("  标签每行: class_id x_center y_center width height  (0~1 归一化)")
    print("  车牌检测通常 class_id=0 表示 plate")
    print()
    print("推荐数据集: CCPD (含车牌四点/矩形，需转为 YOLO)")
    print("  https://github.com/detectRecog/CCPD")
    print()
    print("推荐标注工具:")
    print("  - LabelImg (YOLO 格式导出): https://github.com/HumanSignal/labelImg")
    print("  - Roboflow (在线标注+导出 YOLO)")
    print("  - CVAT: https://www.cvat.ai/")
    print("  - makesense.ai (浏览器免费标注)")
    print()
    print("CCPD 转 YOLO 可参考 CCPD 仓库脚本，或使用:")
    print("  python scripts/import_plate_dataset.py --src <含images+labels的YOLO目录>")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="检测并导入车牌 YOLO 数据集")
    parser.add_argument("--src", type=str, default=str(DEFAULT_PLATE_SRC))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--analyze-only", action="store_true", help="仅分析不导入")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"错误: 路径不存在 {src}")
        sys.exit(1)

    dtype, info = detect_dataset_type(src)
    print(f"数据集类型: {dtype}")
    print(info)

    if dtype == "char_classification":
        print_char_classification_help(info)
        sys.exit(2)

    if dtype != "yolo":
        print(f"无法识别数据集格式: {src}")
        print("请提供 YOLO 格式目录（images/ + labels/）或使用 CCPD 转换后的数据。")
        sys.exit(1)

    if args.analyze_only:
        print("YOLO 格式数据集，可导入。")
        return

    stats = import_plate_yolo(src, val_ratio=args.val_ratio)
    print("\n=== 车牌 YOLO 数据集导入完成 ===")
    print(stats)
    print(f"data.yaml: {DATASETS / 'plate' / 'data.yaml'}")
    print("\n训练示例:")
    print("  python scripts/download_pretrained_weights.py")
    print("  python -m src.train.trainer --task plate --yolo yolov8 --model-size n --epochs 20")


if __name__ == "__main__":
    main()
