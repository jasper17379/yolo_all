#!/usr/bin/env python3
"""评测车牌 OCR 准确率（对比 recognition/*.json 真值）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.third_party_paths import bootstrap_env
from src.tasks.plate_ocr import normalize_plate_text, ocr_on_crop
from src.tasks.plate_recognition import PlateRecognizer


def load_gt(rec_dir: Path) -> list[dict]:
    items = []
    for jf in sorted(rec_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        img_name = data.get("image") or f"{jf.stem}.jpg"
        for p in data.get("plates", []):
            text = p.get("plate_text", "")
            if not text or text == "待填写":
                continue
            items.append({"image": img_name, "json": jf.name, "plate_text": normalize_plate_text(text), "bbox_yolo": p.get("bbox_yolo")})
    return items


def yolo_to_xyxy(bbox_yolo: list, iw: int, ih: int) -> tuple[int, int, int, int]:
    _, cx, cy, w, h = bbox_yolo
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)
    return max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)


def main():
    bootstrap_env()
    parser = argparse.ArgumentParser(description="车牌 OCR 评测")
    parser.add_argument("--split", default="val", choices=["train", "val", "reference"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ocr-engine", default="auto", choices=["auto", "hyperlpr3", "paddleocr"])
    parser.add_argument("--mode", default="crop", choices=["crop", "e2e"], help="crop=按真值框裁剪后 OCR；e2e=PlateRecognizer 全流程")
    parser.add_argument("--yolo", default="yolov8")
    parser.add_argument("--model-size", default="n")
    args = parser.parse_args()

    base = PROJECT_ROOT / "datasets" / "plate"
    if args.split == "reference":
        img_dir = base / "reference" / "images"
        rec_dir = base / "reference" / "recognition"
    else:
        img_dir = base / "images" / args.split
        rec_dir = base / "recognition" / args.split

    gt_list = load_gt(rec_dir)
    if not gt_list:
        print(f"未找到有效识别真值: {rec_dir}")
        return

    engines: dict = {}
    recognizer = PlateRecognizer(yolo_version=args.yolo, model_size=args.model_size, device=args.device, ocr_engine=args.ocr_engine)

    correct = 0
    total = 0
    for gt in gt_list:
        img_path = img_dir / gt["image"]
        if not img_path.exists():
            img_path = base / "images" / args.split / gt["image"]
        if not img_path.exists():
            print(f"  [skip] 缺图 {gt['image']}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        total += 1

        if args.mode == "e2e":
            hits = recognizer.recognize_frame(img, conf=0.01)
            pred = normalize_plate_text(hits[0]["plate_text"]) if hits else ""
        else:
            bbox = gt.get("bbox_yolo")
            if not bbox:
                print(f"  [skip] {gt['image']} 无 bbox_yolo")
                total -= 1
                continue
            x1, y1, x2, y2 = yolo_to_xyxy(bbox, iw, ih)
            crop = img[y1:y2, x1:x2]
            hit = ocr_on_crop(args.ocr_engine, args.device, crop, engines)
            pred = normalize_plate_text(hit.text) if hit else ""

        ok = pred == gt["plate_text"]
        correct += int(ok)
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {gt['image']}: pred={pred or '(空)'} gt={gt['plate_text']}")

    acc = correct / max(total, 1)
    print(f"\n准确率: {correct}/{total} = {acc:.1%} | mode={args.mode} engine={args.ocr_engine}")


if __name__ == "__main__":
    main()
