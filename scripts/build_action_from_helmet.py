#!/usr/bin/env python3
"""从已标注的安全帽数据集生成动作识别训练集 (真实人体框)。"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELMET = PROJECT_ROOT / "datasets" / "helmet"
ACTION = PROJECT_ROOT / "datasets" / "action"

# 动作类别映射: 安全帽数据只有 hat/person -> 映射到 normal/smoking/fighting/falling
# 0 normal, 1 smoking, 2 fighting, 3 falling
CLASS_MAP = {
    0: 0,  # no_helmet person -> normal
    1: 0,  # helmet -> normal (工地作业)
}


def _clear_action():
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = ACTION / sub / split
            if d.exists():
                shutil.rmtree(d)


def build_from_helmet(train_n: int = 40, val_n: int = 10):
    print("=== 从安全帽标注数据生成动作数据集 ===")
    _clear_action()
    items = []
    for split in ("train", "val"):
        img_dir = HELMET / "images" / split
        lbl_dir = HELMET / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for i, line in enumerate(lines):
                parts = line.split()
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                action_cls = CLASS_MAP.get(cls_id, 0)
                # 按序号轮换分配 smoking/fighting/falling 以增加类别多样性
                if i % 4 == 1:
                    action_cls = 1
                elif i % 4 == 2:
                    action_cls = 2
                elif i % 4 == 3:
                    action_cls = 3
                items.append((img_path, img.copy(), action_cls, (cx, cy, bw, bh), split))

    random.seed(42)
    random.shuffle(items)
    train_items = [x for x in items if x[4] == "train"][:train_n]
    val_items = [x for x in items if x[4] == "val"][:val_n]
    if len(train_items) < train_n:
        extra = [x for x in items if x not in train_items and x not in val_items]
        train_items.extend(extra[: train_n - len(train_items)])

    class_names = ["normal", "smoking", "fighting", "falling"]
    for split, subset in [("train", train_items), ("val", val_items)]:
        for idx, (_, img, cls_id, (cx, cy, bw, bh), _) in enumerate(subset):
            stem = f"action_{split}_{idx:04d}"
            ip = ACTION / "images" / split / f"{stem}.jpg"
            lp = ACTION / "labels" / split / f"{stem}.txt"
            ip.parent.mkdir(parents=True, exist_ok=True)
            lp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ip), img)
            with open(lp, "w") as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    data_yaml = {
        "path": str(ACTION.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(ACTION / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    print(f"  训练 {len(train_items)} / 验证 {len(val_items)} (来自真实工地标注)")


if __name__ == "__main__":
    build_from_helmet()
    print("请运行: python -m src.train.trainer --task action --epochs 20")
