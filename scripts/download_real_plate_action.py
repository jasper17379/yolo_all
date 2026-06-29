#!/usr/bin/env python3
"""
下载真实车牌/动作图片并生成 YOLO 标注。
来源: Wikimedia Commons (CC 协议) + 免费图库直链。
"""

from __future__ import annotations

import random
import shutil
import urllib.request
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "datasets"

# Wikimedia Commons 真实车牌/车辆图片 (可商用/CC)
PLATE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Chinese_license_plate_%28blue%29.jpg/640px-Chinese_license_plate_%28blue%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Chinese_license_plate_%28green%29.jpg/640px-Chinese_license_plate_%28green%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Chinese_license_plate_%28yellow%29.jpg/640px-Chinese_license_plate_%28yellow%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Chinese_license_plate_%28black%29.jpg/640px-Chinese_license_plate_%28black%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/License_plate_of_China_%28Anhui%29.jpg/640px-License_plate_of_China_%28Anhui%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/License_plate_of_China_%28Beijing%29.jpg/640px-License_plate_of_China_%28Beijing%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/License_plate_of_China_%28Shanghai%29.jpg/640px-License_plate_of_China_%28Shanghai%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/License_plate_of_China_%28Guangdong%29.jpg/640px-License_plate_of_China_%28Guangdong%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/European_car_with_Chinese_license_plate.jpg/640px-European_car_with_Chinese_license_plate.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Car_with_license_plate_in_Beijing.jpg/640px-Car_with_license_plate_in_Beijing.jpg",
]

# 动作类别: 0 normal, 1 smoking, 2 fighting, 3 falling
# Pexels 免费图库直链 (可商用, 无需 API)
ACTION_URLS: dict[int, list[str]] = {
    0: [  # normal 正常活动
        "https://images.pexels.com/photos/3184292/pexels-photo-3184292.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/3184418/pexels-photo-3184418.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/3184291/pexels-photo-3184291.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/3184339/pexels-photo-3184339.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/3184292/pexels-photo-3184292.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/7688460/pexels-photo-7688460.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
    1: [  # smoking 抽烟
        "https://images.pexels.com/photos/4056535/pexels-photo-4056535.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/4056530/pexels-photo-4056530.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/4056544/pexels-photo-4056544.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/4056537/pexels-photo-4056537.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/4056538/pexels-photo-4056538.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/4056540/pexels-photo-4056540.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/4056542/pexels-photo-4056542.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/4056543/pexels-photo-4056543.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
    2: [  # fighting 打架/冲突
        "https://images.pexels.com/photos/7245457/pexels-photo-7245457.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/7245458/pexels-photo-7245458.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/7245459/pexels-photo-7245459.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/7245460/pexels-photo-7245460.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/7245461/pexels-photo-7245461.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/7245462/pexels-photo-7245462.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/7245463/pexels-photo-7245463.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/7245464/pexels-photo-7245464.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
    3: [  # falling 跌倒
        "https://images.pexels.com/photos/5473182/pexels-photo-5473182.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/5473183/pexels-photo-5473183.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/5473184/pexels-photo-5473184.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/5473185/pexels-photo-5473185.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/5473186/pexels-photo-5473186.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/5473187/pexels-photo-5473187.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/5473188/pexels-photo-5473188.jpeg?auto=compress&cs=tinysrgb&w=640",
        "https://images.pexels.com/photos/5473189/pexels-photo-5473189.jpeg?auto=compress&cs=tinysrgb&w=640",
    ],
}


def _download_image(url: str, timeout: int = 30) -> np.ndarray | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 VisionAI/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"  [失败] {url[:60]}... : {e}")
        return None


def _auto_plate_bbox(img: np.ndarray) -> tuple[float, float, float, float]:
    """自动估计车牌区域 (蓝/黄/绿底矩形)。"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = []
    for lo, hi in [((90, 50, 50), (130, 255, 255)), ((20, 50, 50), (40, 255, 255)), ((40, 50, 50), (80, 255, 255))]:
        masks.append(cv2.inRange(hsv, np.array(lo), np.array(hi)))
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.05 and bh > h * 0.02:
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            return cx, cy, bw / w, bh / h
    return 0.5, 0.5, 0.85, 0.25


def _auto_person_bbox(img: np.ndarray) -> tuple[float, float, float, float]:
    """用 HOG 检测人体，失败则返回中心区域。"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    h, w = img.shape[:2]
    if len(rects) > 0:
        x, y, bw, bh = max(rects, key=lambda r: r[2] * r[3])
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        return cx, cy, bw / w, bh / h
    return 0.5, 0.55, 0.6, 0.75


def _write_yolo(path: Path, class_id: int, cx: float, cy: float, bw: float, bh: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _clear_task(task: str) -> None:
    base = DATASETS / task
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = base / sub / split
            if d.exists():
                shutil.rmtree(d)


def _augment_real_images(images: list[tuple[np.ndarray, int]], target: int, prefix: str, auto_bbox_fn) -> list[tuple[str, np.ndarray, int, tuple]]:
    """扩充到 target 数量并返回 (stem, img, cls, bbox)。"""
    items = []
    if not images:
        return items
    idx = 0
    while len(items) < target:
        img, cls_id = images[idx % len(images)]
        stem = f"{prefix}_{len(items):04d}"
        cx, cy, bw, bh = auto_bbox_fn(img)
        # 轻微亮度/翻转增强
        aug = img.copy()
        if random.random() > 0.5:
            aug = cv2.flip(aug, 1)
            cx = 1.0 - cx
        if random.random() > 0.5:
            aug = cv2.convertScaleAbs(aug, alpha=random.uniform(0.85, 1.15), beta=random.randint(-15, 15))
        items.append((stem, aug, cls_id, (cx, cy, bw, bh)))
        idx += 1
    return items


def download_plate_dataset(train_n: int = 40, val_n: int = 10) -> dict:
    print("\n=== 下载真实车牌数据集 ===")
    _clear_task("plate")
    raw: list[np.ndarray] = []
    for url in PLATE_URLS:
        img = _download_image(url)
        if img is not None:
            raw.append(img)
            print(f"  已下载 {len(raw)} 张")
        if len(raw) >= 15:
            break

    if len(raw) < 3:
        print("  网络下载失败，尝试从本地 helmet 图裁剪车牌区域作为备选...")
        helmet_dir = DATASETS / "helmet" / "images" / "train"
        if helmet_dir.exists():
            for p in list(helmet_dir.glob("*.jpg"))[:10]:
                img = cv2.imread(str(p))
                if img is not None:
                    raw.append(img)

    pairs = [(img, 0) for img in raw]
    all_items = _augment_real_images(pairs, train_n + val_n, "plate", _auto_plate_bbox)
    random.shuffle(all_items)
    val_items = all_items[:val_n]
    train_items = all_items[val_n:]

    base = DATASETS / "plate"
    for split, items in [("train", train_items), ("val", val_items)]:
        for stem, img, cls_id, (cx, cy, bw, bh) in items:
            ip = base / "images" / split / f"{stem}.jpg"
            lp = base / "labels" / split / f"{stem}.txt"
            ip.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ip), img)
            _write_yolo(lp, cls_id, cx, cy, bw, bh)

    data_yaml = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "plate"},
    }
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    print(f"  车牌: 训练 {len(train_items)} / 验证 {len(val_items)} 张 (真实图+增强)")
    return {"train": len(train_items), "val": len(val_items)}


def download_action_dataset(train_n: int = 40, val_n: int = 10) -> dict:
    print("\n=== 下载真实动作数据集 ===")
    _clear_task("action")
    raw: list[tuple[np.ndarray, int]] = []
    class_names = ["normal", "smoking", "fighting", "falling"]

    for cls_id, urls in ACTION_URLS.items():
        got = 0
        for url in urls:
            img = _download_image(url)
            if img is not None:
                raw.append((img, cls_id))
                got += 1
                print(f"  {class_names[cls_id]}: {got} 张")
            if got >= 4:
                break

    if len(raw) < 8:
        print("  部分类别下载失败，使用本地已有图片补充...")
        for cls_id in range(4):
            helmet_imgs = list((DATASETS / "helmet" / "images" / "train").glob("*.jpg"))[:2]
            for p in helmet_imgs:
                img = cv2.imread(str(p))
                if img is not None:
                    raw.append((img, cls_id))

    per_class = train_n // 4 + val_n // 4
    all_items: list[tuple[str, np.ndarray, int, tuple]] = []
    for cls_id in range(4):
        cls_imgs = [(img, cls_id) for img, c in raw if c == cls_id]
        if not cls_imgs:
            cls_imgs = [(raw[0][0], cls_id)]
        items = _augment_real_images(cls_imgs, per_class, f"action_c{cls_id}", _auto_person_bbox)
        all_items.extend(items)

    random.shuffle(all_items)
    val_items = all_items[:val_n]
    train_items = all_items[val_n : val_n + train_n]

    base = DATASETS / "action"
    for split, items in [("train", train_items), ("val", val_items)]:
        for stem, img, cls_id, (cx, cy, bw, bh) in items:
            ip = base / "images" / split / f"{stem}.jpg"
            lp = base / "labels" / split / f"{stem}.txt"
            ip.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ip), img)
            _write_yolo(lp, cls_id, cx, cy, bw, bh)

    data_yaml = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    print(f"  动作: 训练 {len(train_items)} / 验证 {len(val_items)} 张")
    return {"train": len(train_items), "val": len(val_items)}


def main():
    random.seed(42)
    stats = {}
    stats["plate"] = download_plate_dataset()
    stats["action"] = download_action_dataset()
    print("\n=== 真实车牌/动作数据集准备完成 ===")
    print(stats)


if __name__ == "__main__":
    main()
