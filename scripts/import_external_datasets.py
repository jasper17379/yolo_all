#!/usr/bin/env python3
"""从外部路径导入真实数据集，替换合成演示数据。"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "datasets"

# 默认外部数据源
DEFAULT_HELMET_SRC = Path(r"E:\iVS-100-DB-Bak\datasets")
DEFAULT_FACE_SRC = Path(r"E:\iVS-100-DB-Bak\face\face\test")
DEFAULT_PLATE_SRC = Path(r"E:\iVS-100-DB-Bak\datasets\plate")


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_pair(stems: list[str], src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path, prefix: str) -> int:
    count = 0
    for i, stem in enumerate(stems):
        img_src = src_img / f"{stem}.jpg"
        if not img_src.exists():
            for ext in (".jpeg", ".png", ".JPG", ".JPEG"):
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


def import_helmet(src: Path, val_ratio: float = 0.2, seed: int = 42) -> dict:
    """导入安全帽 YOLO 数据集 (class 0=no_helmet, 1=helmet/hat)。"""
    print(f"\n=== 导入安全帽数据集: {src} ===")
    src_img = src / "images"
    src_lbl = src / "labels"
    if not src_img.exists():
        raise FileNotFoundError(f"图像目录不存在: {src_img}")

    stems = sorted(
        p.stem
        for p in src_img.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        and (src_lbl / f"{p.stem}.txt").exists()
    )
    if not stems:
        raise FileNotFoundError(f"未找到匹配的图像/标签: {src}")

    random.seed(seed)
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * val_ratio))
    val_stems = stems[:n_val]
    train_stems = stems[n_val:]

    base = DATASETS / "helmet"
    for split in ("train", "val"):
        _clear_dir(base / "images" / split)
        _clear_dir(base / "labels" / split)

    n_train = _copy_pair(train_stems, src_img, src_lbl, base / "images" / "train", base / "labels" / "train", "helmet_train")
    n_val = _copy_pair(val_stems, src_img, src_lbl, base / "images" / "val", base / "labels" / "val", "helmet_val")

    data_yaml = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "no_helmet", 1: "helmet"},
    }
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)

    print(f"  训练: {n_train} 张, 验证: {n_val} 张 (共 {len(stems)} 张真实工地安全帽图片)")
    return {"train": n_train, "val": n_val, "total": len(stems)}


def import_face(face_src: Path) -> dict:
    """导入人脸数据集: face_src/人名/*.jpg"""
    print(f"\n=== 导入人脸数据集: {face_src} ===")
    if not face_src.exists():
        raise FileNotFoundError(f"人脸目录不存在: {face_src}")

    lfw_dir = DATASETS / "face" / "lfw"
    gallery_dir = DATASETS / "face" / "gallery"
    _clear_dir(lfw_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for f in gallery_dir.glob("embeddings.pkl"):
        f.unlink()
    for d in gallery_dir.iterdir():
        if d.is_dir():
            shutil.rmtree(d)

    img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    person_count = 0
    img_count = 0

    for person_dir in sorted(face_src.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        dst_dir = lfw_dir / person_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in sorted(person_dir.iterdir()):
            if src.suffix.lower() not in img_ext:
                continue
            dest = dst_dir / src.name
            shutil.copy2(src, dest)
            copied += 1
            img_count += 1
        if copied:
            person_count += 1
            print(f"  {person_name}: {copied} 张")

    if person_count == 0:
        raise FileNotFoundError(f"未找到任何人脸子目录: {face_src}")

    print(f"  共 {person_count} 人, {img_count} 张 -> {lfw_dir}")
    return {"persons": person_count, "images": img_count}


def main():
    parser = argparse.ArgumentParser(description="导入外部真实数据集")
    parser.add_argument("--helmet-src", type=str, default=str(DEFAULT_HELMET_SRC))
    parser.add_argument("--face-src", type=str, default=str(DEFAULT_FACE_SRC))
    parser.add_argument("--plate-src", type=str, default=str(DEFAULT_PLATE_SRC), help="YOLO 格式 plate 数据(images+labels)")
    parser.add_argument("--skip-helmet", action="store_true")
    parser.add_argument("--skip-face", action="store_true")
    parser.add_argument("--skip-plate", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    helmet_src = Path(args.helmet_src)
    face_src = Path(args.face_src)

    stats = {}
    if not args.skip_helmet:
        stats["helmet"] = import_helmet(helmet_src, val_ratio=args.val_ratio)
    if not args.skip_face:
        stats["face"] = import_face(face_src)
    if not args.skip_plate:
        import importlib.util

        plate_mod_path = Path(__file__).parent / "import_plate_dataset.py"
        spec = importlib.util.spec_from_file_location("import_plate_dataset", plate_mod_path)
        plate_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(plate_mod)

        plate_src = Path(args.plate_src)
        if plate_src.exists():
            dtype, info = plate_mod.detect_dataset_type(plate_src)
            if dtype == "yolo":
                stats["plate"] = plate_mod.import_plate_yolo(plate_src, val_ratio=args.val_ratio)
            elif dtype == "char_classification":
                plate_mod.print_char_classification_help(info)
                print("\n跳过 plate 导入。请使用 YOLO 检测数据或运行: python scripts/import_plate_dataset.py --analyze-only")
            else:
                print(f"跳过 plate: 无法识别格式 {plate_src}")

    print("\n=== 真实数据集导入完成 ===")
    print(stats)


if __name__ == "__main__":
    main()
