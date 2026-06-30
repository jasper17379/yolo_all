#!/usr/bin/env python3
"""导出模型为 ONNX / RKNN 等部署格式"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.device import add_device_arg, resolve_yolo_device
from src.core.weights import get_task_weights
from src.core.yolo_wrapper import YOLOWrapper


def export_task(
    task: str,
    fmt: str = "onnx",
    yolo_version: str = "yolov8",
    model_size: str = "n",
    device: str = "auto",
    imgsz: int = 640,
) -> str:
    weights = get_task_weights(task, yolo_version, model_size)
    wrapper = YOLOWrapper(version=yolo_version, weights=weights, model_size=model_size)
    yolo_device = resolve_yolo_device(device)
    path = wrapper.export(fmt=fmt, imgsz=imgsz, device=yolo_device)
    print(f"[{task}] 权重: {weights}")
    print(f"[{task}] 导出 {fmt}: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["helmet", "plate", "action", "all"])
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "rknn"])
    parser.add_argument("--yolo", default="yolov8", choices=["yolov5", "yolov8", "yolov10"])
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x", "b"])
    add_device_arg(parser)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    tasks = ["helmet", "plate", "action"] if args.task == "all" else [args.task]
    for t in tasks:
        try:
            export_task(t, args.format, args.yolo, args.model_size, args.device, args.imgsz)
        except Exception as e:
            print(f"[{t}] 导出失败: {e}")


if __name__ == "__main__":
    main()
