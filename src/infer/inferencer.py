"""统一推理入口"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from src.core.config import PROJECT_ROOT, load_task_config, resolve_path
from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper


def get_task_weights(task: str) -> Path:
    """获取任务权重，优先 weights/{task}/best.pt。"""
    custom = PROJECT_ROOT / "weights" / task / "best.pt"
    if custom.exists():
        return custom
    runs = sorted(
        (PROJECT_ROOT / "runs" / "train" / task).glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if runs:
        return runs[0]
    cfg = load_task_config(task)
    return Path(cfg.get("default_weights", "yolov8n.pt"))


def infer_detection(
    task: str,
    source: str | Path,
    yolo_version: YOLOVersion = "yolov8",
    conf: float = 0.25,
    save: bool = True,
) -> list[dict[str, Any]]:
    weights = get_task_weights(task)
    wrapper = YOLOWrapper(version=yolo_version, weights=weights)
    results = wrapper.predict(
        source=source,
        conf=conf,
        project=str(PROJECT_ROOT / "runs" / "predict"),
        name=task,
        save=save,
    )
    outputs = []
    for r in results:
        item: dict[str, Any] = {"task": task, "path": getattr(r, "path", str(source)), "detections": []}
        if r.boxes is not None:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                item["detections"].append(
                    {
                        "class": names[cls_id],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )
        outputs.append(item)
    return outputs


def infer_face(source: str | Path, threshold: float = 0.4) -> list[dict[str, Any]]:
    from src.tasks.face_recognition import FaceRecognizer

    rec = FaceRecognizer()
    img = cv2.imread(str(source))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {source}")
    faces = rec.recognize(img, threshold=threshold)
    return [{"task": "face", "path": str(source), "faces": faces}]


def infer_plate(source: str | Path, yolo_version: YOLOVersion = "yolov8", conf: float = 0.25) -> list[dict]:
    from src.tasks.plate_recognition import PlateRecognizer

    rec = PlateRecognizer(yolo_version=yolo_version)
    return rec.recognize(source, conf=conf)


def infer_task(
    task: str,
    source: str | Path,
    yolo_version: YOLOVersion = "yolov8",
    conf: float = 0.25,
    save: bool = True,
) -> list[dict[str, Any]]:
    cfg = load_task_config(task)
    if cfg.get("type") == "recognition":
        return infer_face(source, threshold=cfg.get("rec_threshold", 0.4))
    if task == "plate":
        return infer_plate(source, yolo_version=yolo_version, conf=conf)
    return infer_detection(task, source, yolo_version, conf, save)


def main():
    parser = argparse.ArgumentParser(description="Vision AI 统一推理")
    parser.add_argument("--task", required=True, choices=["helmet", "plate", "action", "face"])
    parser.add_argument("--source", required=True, help="图像/目录/视频路径")
    parser.add_argument("--yolo", default="yolov8", choices=["yolov5", "yolov8", "yolov10"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    results = infer_task(
        task=args.task,
        source=resolve_path(args.source),
        yolo_version=args.yolo,
        conf=args.conf,
        save=not args.no_save,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
