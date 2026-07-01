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
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = PROJECT_ROOT / "datasets"

_PLATE_FONT: ImageFont.FreeTypeFont | None = None


def _plate_font(size: int = 28) -> ImageFont.FreeTypeFont:
    global _PLATE_FONT
    if _PLATE_FONT is not None:
        return _PLATE_FONT
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for fp in candidates:
        if Path(fp).exists():
            _PLATE_FONT = ImageFont.truetype(fp, size)
            return _PLATE_FONT
    _PLATE_FONT = ImageFont.load_default()
    return _PLATE_FONT


def _put_chinese_text(img: np.ndarray, text: str, xy: tuple[int, int], color=(255, 255, 255), size: int = 28) -> np.ndarray:
    """在 BGR 图上绘制中文（OpenCV putText 不支持中文）。"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, font=_plate_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

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

# 本地生成范例（标注框 + 车牌号均为程序写入，100% 对齐）
LOCAL_PLATE_SAMPLES = [
    {"name": "plate_local_scene_rear", "split": "train", "generator": "scene_rear", "plate_text": "京A12345", "plate_type": "蓝牌"},
    {"name": "plate_local_scene_front", "split": "train", "generator": "scene_front", "plate_text": "沪B88888", "plate_type": "蓝牌"},
    {"name": "plate_local_closeup_blue", "split": "train", "generator": "closeup", "plate_text": "粤C66666", "plate_type": "蓝牌"},
    {"name": "plate_local_scene_rear2", "split": "train", "generator": "scene_rear", "plate_text": "京D98765", "plate_type": "蓝牌"},
    {"name": "plate_local_closeup2", "split": "val", "generator": "closeup", "plate_text": "浙E55555", "plate_type": "蓝牌"},
    {"name": "plate_local_scene_front2", "split": "val", "generator": "scene_front", "plate_text": "苏F77777", "plate_type": "蓝牌"},
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


def _draw_preview(
    img: np.ndarray,
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    plate_text: str = "",
) -> np.ndarray:
    out = img.copy()
    ih, iw = out.shape[:2]
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    tag = plate_text if plate_text else str(class_id)
    from src.core.cv_draw import put_text_bgr

    return put_text_bgr(out, tag, (x1, max(20, y1 - 5)), (0, 255, 0), font_size=18)


def _write_recognition_json(
    path: Path,
    image_name: str,
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    plate_text: str,
    plate_type: str = "蓝牌",
    note: str = "",
) -> None:
    """识别标注侧车文件（与 YOLO 检测标注分开，用于 OCR 评测/核对）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0",
        "image": image_name,
        "description": "车牌识别真值：与 labels/*.txt 检测框一一对应",
        "plates": [
            {
                "plate_text": plate_text,
                "plate_type": plate_type,
                "bbox_yolo": [class_id, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
                "note": note,
            }
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


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
    plate_text: str = "",
    plate_type: str = "蓝牌",
    task: str = "action",
) -> None:
    img_name = f"{name}.jpg"
    img_path = base / "images" / split / img_name
    lbl_path = base / "labels" / split / f"{name}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), img)
    _write_yolo(lbl_path, class_id, cx, cy, w, h)

    ref_img = ref / "images" / img_name
    ref_lbl = ref / "labels" / f"{name}.txt"
    ref_img.parent.mkdir(parents=True, exist_ok=True)
    ref_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref_img), img)
    shutil.copy2(lbl_path, ref_lbl)

    if task == "plate":
        rec_path = base / "recognition" / split / f"{name}.json"
        ref_rec = ref / "recognition" / f"{name}.json"
        text = plate_text or "待填写"
        _write_recognition_json(rec_path, img_name, class_id, cx, cy, w, h, text, plate_type, note)
        ref_rec.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec_path, ref_rec)

    (ref / "preview").mkdir(parents=True, exist_ok=True)
    preview_tag = plate_text if task == "plate" else ""
    cv2.imwrite(
        str(ref / "preview" / f"{name}_preview.jpg"),
        _draw_preview(img, class_id, cx, cy, w, h, preview_tag),
    )
    extra = f" | 车牌={plate_text}" if plate_text else ""
    print(f"  [ok] {name} ({split}) - {note}{extra}")


def _gen_plate_scene_rear(plate_text: str = "京A12345") -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """车尾场景 + 蓝牌，返回图像与 (cx,cy,w,h)。"""
    w, h = 640, 480
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (560, 400), (120, 120, 120), -1)
    px, py, pw, ph = 220, 360, 200, 44
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (200, 80, 30), -1)
    img = _put_chinese_text(img, plate_text, (px + 8, py + 4), size=24)
    cx = (px + pw / 2) / w
    cy = (py + ph / 2) / h
    return img, (cx, cy, pw / w, ph / h)


def _gen_plate_scene_front(plate_text: str = "沪B88888") -> tuple[np.ndarray, tuple[float, float, float, float]]:
    w, h = 640, 480
    img = np.full((h, w, 3), 140, dtype=np.uint8)
    cv2.ellipse(img, (320, 300), (200, 120), 0, 0, 360, (60, 60, 60), -1)
    px, py, pw, ph = 250, 390, 140, 36
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (200, 80, 30), -1)
    img = _put_chinese_text(img, plate_text, (px + 6, py + 2), size=20)
    cx = (px + pw / 2) / w
    cy = (py + ph / 2) / h
    return img, (cx, cy, pw / w, ph / h)


def _gen_plate_closeup(plate_text: str = "粤C66666") -> tuple[np.ndarray, tuple[float, float, float, float]]:
    w, h = 400, 120
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (200, 80, 30)
    img = _put_chinese_text(img, plate_text, (20, 30), size=36)
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
    "scene_rear": lambda text: _gen_plate_scene_rear(text),
    "scene_front": lambda text: _gen_plate_scene_front(text),
    "closeup": lambda text: _gen_plate_closeup(text),
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
        _save_sample(
            base, ref, item["name"], item["split"], img, cls_id, cx, cy, w, h, item["note"],
            plate_text=item.get("plate_text", "待填写"),
            plate_type=item.get("plate_type", "蓝牌"),
            task="plate",
        )
        manifest.append(item)
        import time
        time.sleep(1.2)  # 降低 Wikimedia 429 限流

    for item in LOCAL_PLATE_SAMPLES:
        text = item["plate_text"]
        img, (cx, cy, w, h) = PLATE_GENS[item["generator"]](text)
        _save_sample(
            base, ref, item["name"], item["split"], img, 0, cx, cy, w, h,
            f"本地范例-{item['generator']}",
            plate_text=text,
            plate_type=item.get("plate_type", "蓝牌"),
            task="plate",
        )
        manifest.append({**item, "bbox": [0, cx, cy, w, h], "note": "本地精确标注"})

    train_n = len(list((base / "images" / "train").glob("*.jpg")))
    val_n = len(list((base / "images" / "val").glob("*.jpg")))
    if train_n + val_n < 4:
        raise RuntimeError("车牌范例不足")

    with open(ref / "classes.txt", "w", encoding="utf-8") as f:
        f.write("plate\n")

    templates = ref / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    with open(templates / "detection_label_example.txt", "w", encoding="utf-8") as f:
        f.write("# YOLO 检测标注（LabelImg 导出）\n# 每行: class_id x_center y_center width height (0~1)\n0 0.500000 0.795833 0.312500 0.091667\n")
    with open(templates / "recognition_label_example.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "1.0",
                "image": "your_image.jpg",
                "description": "车牌号真值，与 labels 中检测框一一对应",
                "plates": [
                    {
                        "plate_text": "京A12345",
                        "plate_type": "蓝牌",
                        "bbox_yolo": [0, 0.5, 0.795833, 0.3125, 0.091667],
                        "note": "人工核对车牌号",
                    }
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    with open(ref / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "车牌标注参考（检测与识别分开）\n"
            "================================\n"
            "本项目车牌流程: YOLO 框车牌(可训练) + HyperLPR3 读字(预训练，无需训练)\n\n"
            "一、检测标注（训练 YOLO）\n"
            "  工具: LabelImg，格式 YOLO\n"
            "  目录: reference/images + reference/labels\n"
            "  类别: plate (class_id=0)\n"
            "  每行: 0 cx cy w h (归一化 0~1)\n"
            "  模板: templates/detection_label_example.txt\n\n"
            "二、识别标注（OCR 评测/核对，不用于训练）\n"
            "  目录: reference/recognition/*.json\n"
            "  与检测框一一对应，填写 plate_text 真值\n"
            "  模板: templates/recognition_label_example.json\n\n"
            "三、新增数据\n"
            "  images/train + labels/train + recognition/train (可选)\n"
            "  对照 preview/ 红框与绿字核对\n\n"
            "四、评测 OCR 准确率\n"
            "  python scripts/eval_plate_recognition.py --split val\n"
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
