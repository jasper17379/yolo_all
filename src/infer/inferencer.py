"""
统一推理入口。

示例:
  python -m src.infer.inferencer --task helmet --source datasets/helmet/images/val --device auto
  python -m src.infer.inferencer --task plate --source test.jpg --device 0 --conf 0.3 --imgsz 640
  python -m src.infer.inferencer --task face --source photo.jpg --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from src.core.config import load_global_config, load_task_config, resolve_path
from src.core.device import add_device_arg, print_device_info, resolve_yolo_device
from src.core.third_party_paths import bootstrap_env
from src.core.train_config import InferHyperParams
from src.core.weights import get_task_weights
from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper


def infer_detection(
    task: str,
    source: str | Path,
    yolo_version: YOLOVersion = "yolov8",
    model_size: str = "n",
    device: str = "auto",
    hyper: InferHyperParams | None = None,
    save: bool = True,
) -> list[dict[str, Any]]:
    hyper = hyper or InferHyperParams.from_global()
    weights = get_task_weights(task, yolo_version, model_size)
    wrapper = YOLOWrapper(version=yolo_version, weights=weights, model_size=model_size)
    yolo_device = resolve_yolo_device(device)
    results = wrapper.predict(
        source=source,
        conf=hyper.conf,
        iou=hyper.iou,
        imgsz=hyper.imgsz,
        half=hyper.half,
        device=yolo_device,
        project=str(resolve_path("runs/predict")),
        name=f"{task}_{yolo_version}{model_size}",
        save=save,
    )
    outputs = []
    for r in results:
        item: dict[str, Any] = {
            "task": task,
            "device": str(yolo_device),
            "weights": str(weights),
            "path": getattr(r, "path", str(source)),
            "detections": [],
        }
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


def infer_face(
    source: str | Path,
    threshold: float = 0.4,
    device: str = "auto",
) -> list[dict[str, Any]]:
    from src.tasks.face_recognition import FaceRecognizer

    rec = FaceRecognizer(device=device)
    img = cv2.imread(str(source))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {source}")
    faces = rec.recognize(img, threshold=threshold)
    return [{"task": "face", "path": str(source), "device": device, "faces": faces}]


def infer_plate(
    source: str | Path,
    yolo_version: YOLOVersion = "yolov8",
    model_size: str = "n",
    device: str = "auto",
    hyper: InferHyperParams | None = None,
) -> list[dict]:
    from src.tasks.plate_recognition import PlateRecognizer

    hyper = hyper or InferHyperParams.from_global()
    rec = PlateRecognizer(yolo_version=yolo_version, model_size=model_size, device=device)
    return rec.recognize(source, conf=hyper.conf, imgsz=hyper.imgsz)


def infer_task(
    task: str,
    source: str | Path,
    yolo_version: YOLOVersion = "yolov8",
    model_size: str = "n",
    device: str = "auto",
    hyper: InferHyperParams | None = None,
    save: bool = True,
) -> list[dict[str, Any]]:
    cfg = load_task_config(task)
    hyper = hyper or InferHyperParams.from_global()
    if cfg.get("type") == "recognition":
        return infer_face(source, threshold=cfg.get("rec_threshold", 0.4), device=device)
    if task == "plate":
        return infer_plate(source, yolo_version, model_size, device, hyper)
    return infer_detection(task, source, yolo_version, model_size, device, hyper, save)


def main():
    bootstrap_env()
    g = load_global_config()
    inf = g.get("infer", {})
    parser = argparse.ArgumentParser(description="Vision AI 统一推理")
    parser.add_argument("--task", required=True, choices=["helmet", "plate", "action", "face"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--yolo", default=g.get("default_yolo_version", "yolov8"), choices=["yolov5", "yolov8", "yolov10"])
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x", "b"])
    add_device_arg(parser, default=g.get("device", "auto"))
    parser.add_argument("--conf", type=float, default=inf.get("conf", 0.25))
    parser.add_argument("--iou", type=float, default=inf.get("iou", 0.45))
    parser.add_argument("--imgsz", type=int, default=inf.get("imgsz", 640))
    parser.add_argument("--half", action="store_true", default=bool(inf.get("half", False)), help="FP16 推理(GPU)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    print_device_info(args.device)
    hyper = InferHyperParams.from_global(
        {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "half": args.half}
    )

    results = infer_task(
        task=args.task,
        source=resolve_path(args.source),
        yolo_version=args.yolo,
        model_size=args.model_size,
        device=args.device,
        hyper=hyper,
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
