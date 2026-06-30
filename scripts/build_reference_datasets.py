#!/usr/bin/env python3
"""
构建车牌/动作「参考范例」数据集（手工核对过的 YOLO 标注）。

- 清理旧的自动标注错误数据
- 优先下载 Wikimedia 原图；失败则用本地生成的精确标注范例图
- 生成 reference/ 目录（含 preview 红框、classes.txt、LabelImg 说明）

用法:
  python scripts/build_reference_datasets.py
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "datasets"

# 使用 Commons 原图直链（非 thumb，避免 400）
PLATE_SAMPLES = [
    {
        "name": "plate_close_blue",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4a/Chinese_license_plate_%28blue%29.jpg",
        "split": "train",
        "bbox": [0, 0.500, 0.500, 0.920, 0.820],
        "note": "蓝牌特写",
    },
    {
        "name": "plate_close_green",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Chinese_license_plate_%28green%29.jpg",
        "split": "train",
        "bbox": [0, 0.500, 0.500, 0.900, 0.800],
        "note": "绿牌特写",
    },
    {
        "name": "plate_close_yellow",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Chinese_license_plate_%28yellow%29.jpg",
        "split": "train",
        "bbox": [0, 0.500, 0.500, 0.900, 0.780],
        "note": "黄牌特写",
    },
    {
        "name": "plate_beijing_car",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f5/Car_with_license_plate_in_Beijing.jpg",
        "split": "train",
        "bbox": [0, 0.520, 0.820, 0.220, 0.090],
        "note": "场景图车尾车牌",
    },
    {
        "name": "plate_shanghai",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/0c/License_plate_of_China_%28Shanghai%29.jpg",
        "split": "train",
        "bbox": [0, 0.500, 0.500, 0.880, 0.750],
        "note": "沪牌特写",
    },
    {
        "name": "plate_guangdong",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/6f/License_plate_of_China_%28Guangdong%29.jpg",
        "split": "val",
        "bbox": [0, 0.500, 0.500, 0.850, 0.720],
        "note": "粤牌特写",
    },
    {
        "name": "plate_eu_china_car",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/European_car_with_Chinese_license_plate.jpg",
        "split": "val",
        "bbox": [0, 0.480, 0.750, 0.180, 0.070],
        "note": "场景图前牌",
    },
]

ACTION_SAMPLES = [
    {
        "name": "action_smoking_man",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/14/Smoking_man.jpg",
        "class_id": 1,
        "split": "train",
        "bbox": [1, 0.480, 0.450, 0.420, 0.820],
        "note": "抽烟",
    },
    {
        "name": "action_smoking_cig",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Cigarette_smoking.jpg",
        "class_id": 1,
        "split": "val",
        "bbox": [1, 0.500, 0.500, 0.600, 0.650],
        "note": "手持香烟",
    },
    {
        "name": "action_fighting_boxing",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8f/Boxing_match.jpg",
        "class_id": 2,
        "split": "train",
        "bbox": [2, 0.500, 0.500, 0.700, 0.850],
        "note": "拳击对抗",
    },
    {
        "name": "action_falling_cyclist",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Fallen_cyclist.jpg",
        "class_id": 3,
        "split": "train",
        "bbox": [3, 0.450, 0.620, 0.500, 0.550],
        "note": "骑车跌倒",
    },
]

# 本地生成范例（标注框为程序写入，100% 对齐）
LOCAL_PLATE_SAMPLES = [
    {"name": "plate_local_scene_rear", "split": "train", "generator": "scene_rear"},
    {"name": "plate_local_scene_front", "split": "train", "generator": "scene_front"},
    {"name": "plate_local_closeup_blue", "split": "train", "generator": "closeup"},
    {"name": "plate_local_scene_rear2", "split": "train", "generator": "scene_rear"},
    {"name": "plate_local_closeup2", "split": "val", "generator": "closeup"},
    {"name": "plate_local_scene_front2", "split": "val", "generator": "scene_front"},
]

LOCAL_ACTION_SAMPLES = [
    {"name": "action_local_normal", "class_id": 0, "split": "train", "generator": "normal"},
    {"name": "action_local_normal2", "class_id": 0, "split": "train", "generator": "normal"},
    {"name": "action_local_smoking", "class_id": 1, "split": "train", "generator": "smoking"},
    {"name": "action_local_smoking2", "class_id": 1, "split": "val", "generator": "smoking"},
    {"name": "action_local_fighting", "class_id": 2, "split": "train", "generator": "fighting"},
    {"name": "action_local_fighting2", "class_id": 2, "split": "val", "generator": "fighting"},
    {"name": "action_local_falling", "class_id": 3, "split": "train", "generator": "falling"},
    {"name": "action_local_falling2", "class_id": 3, "split": "val", "generator": "falling"},
]


def _download(url: str) -> np.ndarray | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VisionAI/1.0 (reference dataset builder)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [下载失败] {url[:75]}... : {e}")
        return None


def _clear_task(task: str) -> None:
    base = DATASETS / task
    for sub in ("images", "labels", "reference"):
        p = base / sub
        if p.exists():
            shutil.rmtree(p)


def _write_yolo(label_path: Path, class_id: int, cx: float, cy: float, w: float, h: float) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _draw_preview(img: np.ndarray, class_id: int, cx: float, cy: float, w: float, h: float) -> np.ndarray:
    out = img.copy()
    ih, iw = out.shape[:2]
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(out, str(class_id), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def _save_sample(
    base: Path,
    ref: Path,
    name: str,
    split: str,
    img: np.ndarray,
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    note: str,
) -> None:
    img_path = base / "images" / split / f"{name}.jpg"
    lbl_path = base / "labels" / split / f"{name}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), img)
    _write_yolo(lbl_path, class_id, cx, cy, w, h)

    ref_img = ref / "images" / f"{name}.jpg"
    ref_lbl = ref / "labels" / f"{name}.txt"
    ref_img.parent.mkdir(parents=True, exist_ok=True)
    ref_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref_img), img)
    shutil.copy2(lbl_path, ref_lbl)
    (ref / "preview").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref / "preview" / f"{name}_preview.jpg"), _draw_preview(img, class_id, cx, cy, w, h))
    print(f"  [ok] {name} ({split}) - {note}")


def _gen_plate_scene_rear() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """车尾场景 + 蓝牌，返回图像与 (cx,cy,w,h)。"""
    w, h = 640, 480
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (560, 400), (120, 120, 120), -1)  # 车身
    px, py, pw, ph = 220, 360, 200, 44
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (200, 80, 30), -1)  # 蓝牌
    cv2.putText(img, "Jing A12345", (px + 8, py + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cx = (px + pw / 2) / w
    cy = (py + ph / 2) / h
    return img, (cx, cy, pw / w, ph / h)


def _gen_plate_scene_front() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    w, h = 640, 480
    img = np.full((h, w, 3), 140, dtype=np.uint8)
    cv2.ellipse(img, (320, 300), (200, 120), 0, 0, 360, (60, 60, 60), -1)
    px, py, pw, ph = 250, 390, 140, 36
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (200, 80, 30), -1)
    cv2.putText(img, "Shanghai", (px + 6, py + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cx = (px + pw / 2) / w
    cy = (py + ph / 2) / h
    return img, (cx, cy, pw / w, ph / h)


def _gen_plate_closeup() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    w, h = 400, 120
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (200, 80, 30)
    cv2.putText(img, "Plate Demo", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return img, (0.5, 0.5, 0.92, 0.85)


def _gen_action(kind: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    w, h = 480, 640
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    # 人体区域固定，便于标注
    bx, by, bw, bh = 140, 100, 200, 420
    body_color = (180, 160, 140)
    cv2.rectangle(img, (bx, by + 80), (bx + bw, by + bh), body_color, -1)  # 躯干
    cv2.circle(img, (bx + bw // 2, by + 50), 45, body_color, -1)  # 头

    if kind == "normal":
        cv2.line(img, (bx + 20, by + 120), (bx - 30, by + 200), (100, 100, 100), 8)
        cv2.line(img, (bx + bw - 20, by + 120), (bx + bw + 30, by + 200), (100, 100, 100), 8)
        note = "正常站立"
    elif kind == "smoking":
        cv2.line(img, (bx + bw - 10, by + 150), (bx + bw + 40, by + 130), (80, 80, 80), 6)
        cv2.circle(img, (bx + bw + 45, by + 125), 5, (50, 50, 50), -1)
        cv2.line(img, (bx + bw + 45, by + 120), (bx + bw + 45, by + 100), (200, 200, 200), 2)
        note = "抽烟"
    elif kind == "fighting":
        cv2.rectangle(img, (300, 120), (420, 500), (160, 140, 120), -1)
        cv2.circle(img, (360, 90), 40, (160, 140, 120), -1)
        cv2.line(img, (bx + bw, by + 180), (300, by + 200), (0, 0, 200), 6)
        note = "双人对抗"
        bx, by, bw, bh = 120, 90, 320, 430
    elif kind == "falling":
        img = np.full((h, w, 3), 200, dtype=np.uint8)
        bx, by, bw, bh = 100, 320, 280, 120
        cv2.ellipse(img, (240, 380), (140, 50), 15, 0, 360, body_color, -1)
        cv2.circle(img, (130, 350), 35, body_color, -1)
        note = "跌倒"
    else:
        note = kind

    cx = (bx + bw / 2) / w
    cy = (by + bh / 2) / h
    return img, (cx, cy, bw / w, bh / h)


PLATE_GENS = {
    "scene_rear": _gen_plate_scene_rear,
    "scene_front": _gen_plate_scene_front,
    "closeup": _gen_plate_closeup,
}
ACTION_GENS = {
    "normal": lambda: _gen_action("normal"),
    "smoking": lambda: _gen_action("smoking"),
    "fighting": lambda: _gen_action("fighting"),
    "falling": lambda: _gen_action("falling"),
}


def _build_plate() -> dict:
    print("\n=== 重建车牌参考数据集 ===")
    _clear_task("plate")
    base = DATASETS / "plate"
    ref = base / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    manifest = []

    for item in PLATE_SAMPLES:
        img = _download(item["url"])
        if img is None:
            continue
        cls_id, cx, cy, w, h = item["bbox"]
        _save_sample(base, ref, item["name"], item["split"], img, cls_id, cx, cy, w, h, item["note"])
        manifest.append(item)
        import time
        time.sleep(1.2)  # 降低 Wikimedia 429 限流

    for item in LOCAL_PLATE_SAMPLES:
        img, (cx, cy, w, h) = PLATE_GENS[item["generator"]]()
        _save_sample(base, ref, item["name"], item["split"], img, 0, cx, cy, w, h, f"本地范例-{item['generator']}")
        manifest.append({**item, "bbox": [0, cx, cy, w, h], "note": "本地精确标注"})

    train_n = len(list((base / "images" / "train").glob("*.jpg")))
    val_n = len(list((base / "images" / "val").glob("*.jpg")))
    if train_n + val_n < 4:
        raise RuntimeError("车牌范例不足")

    with open(ref / "classes.txt", "w", encoding="utf-8") as f:
        f.write("plate\n")
    with open(ref / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "车牌 YOLO 标注参考\n"
            "1. LabelImg 打开 reference/images，格式 YOLO，类别 plate\n"
            "2. 对照 preview/ 红框\n"
            "3. 新数据放入 datasets/plate/images/train + labels/train\n"
            "4. 每行标签: 0 cx cy w h (0~1)\n"
        )
    data_yaml = {"path": str(base.resolve()), "train": "images/train", "val": "images/val", "nc": 1, "names": {0: "plate"}}
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    with open(ref / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)
    return {"train": train_n, "val": val_n, "reference": str(ref)}


def _build_action() -> dict:
    print("\n=== 重建动作参考数据集 ===")
    _clear_task("action")
    base = DATASETS / "action"
    ref = base / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    class_names = ["normal", "smoking", "fighting", "falling"]
    manifest = []

    for item in ACTION_SAMPLES:
        img = _download(item["url"])
        if img is None:
            continue
        cls_id, cx, cy, w, h = item["bbox"]
        _save_sample(base, ref, item["name"], item["split"], img, cls_id, cx, cy, w, h, item["note"])
        manifest.append(item)

    for item in LOCAL_ACTION_SAMPLES:
        img, (cx, cy, w, h) = ACTION_GENS[item["generator"]]()
        cid = item["class_id"]
        _save_sample(base, ref, item["name"], item["split"], img, cid, cx, cy, w, h, f"本地范例-{class_names[cid]}")
        manifest.append({**item, "bbox": [cid, cx, cy, w, h]})

    train_n = len(list((base / "images" / "train").glob("*.jpg")))
    val_n = len(list((base / "images" / "val").glob("*.jpg")))
    if train_n + val_n < 4:
        raise RuntimeError("动作范例不足")

    with open(ref / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")
    with open(ref / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "动作 YOLO 标注参考\n"
            "0 normal  1 smoking  2 fighting  3 falling\n"
            "框住发生该动作的人体区域。新增数据放入 images/train + labels/train\n"
        )
    data_yaml = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    with open(ref / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)
    return {"train": train_n, "val": val_n, "reference": str(ref)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["plate", "action", "all"], default="all")
    args = parser.parse_args()
    stats = {}
    if args.task in ("plate", "all"):
        stats["plate"] = _build_plate()
    if args.task in ("action", "all"):
        stats["action"] = _build_action()
    print("\n=== 参考范例数据集构建完成 ===")
    print(stats)
    print("\n核对: datasets/{plate,action}/reference/preview/")


if __name__ == "__main__":
    main()
