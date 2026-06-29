#!/usr/bin/env python3
"""导出模型为 ONNX / RKNN 等部署格式"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.yolo_wrapper import YOLOWrapper
from src.infer.inferencer import get_task_weights


def export_task(task: str, fmt: str = "onnx", yolo_version: str = "yolov8") -> str:
    weights = get_task_weights(task)
    wrapper = YOLOWrapper(version=yolo_version, weights=weights)
    out_dir = PROJECT_ROOT / "weights" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    path = wrapper.export(fmt=fmt, imgsz=640)
    print(f"[{task}] 导出 {fmt}: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["helmet", "plate", "action", "all"])
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "rknn"])
    parser.add_argument("--yolo", default="yolov8")
    args = parser.parse_args()

    tasks = ["helmet", "plate", "action"] if args.task == "all" else [args.task]
    for t in tasks:
        try:
            export_task(t, args.format, args.yolo)
        except Exception as e:
            print(f"[{t}] 导出失败: {e}")


if __name__ == "__main__":
    main()
