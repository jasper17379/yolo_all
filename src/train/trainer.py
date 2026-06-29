"""统一训练入口"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.core.config import PROJECT_ROOT, ensure_dirs, load_global_config, load_task_config, resolve_path
from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper


def find_best_weights(task: str) -> Path | None:
    """查找上次训练的最佳权重。"""
    runs_dir = PROJECT_ROOT / "runs" / "train" / task
    if not runs_dir.exists():
        return None
    candidates = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def train_task(
    task: str,
    yolo_version: YOLOVersion = "yolov8",
    epochs: int = 20,
    batch: int = 8,
    imgsz: int = 640,
    resume_from_best: bool = False,
    weights: str | None = None,
    resume: bool = False,
) -> dict:
    """训练指定任务。"""
    ensure_dirs()
    task_cfg = load_task_config(task)
    global_cfg = load_global_config()

    if task_cfg.get("type") == "recognition":
        from src.tasks.face_trainer import train_face_gallery

        return train_face_gallery()

    data_yaml = resolve_path(task_cfg["dataset"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集配置不存在: {data_yaml}，请先运行 python scripts/download_datasets.py")

    default_w = weights or task_cfg.get("default_weights") or global_cfg.get("default_weights", "yolov8n.pt")
    best_path = find_best_weights(task)

    if resume_from_best and best_path:
        wrapper = YOLOWrapper.from_best_or_pretrained(yolo_version, best_path, default_w)
        print(f"[{task}] 基于上次 best 权重继续训练: {best_path}")
    else:
        wrapper = YOLOWrapper(version=yolo_version, weights=default_w)
        print(f"[{task}] 从预训练权重开始训练: {default_w}")

    result = wrapper.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=task,
        resume=resume,
    )

    # 同步 best 到 weights 目录
    dest = PROJECT_ROOT / "weights" / task / "best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result["best"], dest)
    print(f"[{task}] 训练完成, best 权重: {dest}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Vision AI 统一训练")
    parser.add_argument("--task", required=True, choices=["helmet", "plate", "action", "face", "all"])
    parser.add_argument("--yolo", default="yolov8", choices=["yolov5", "yolov8", "yolov10"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume-from-best", action="store_true", help="在上次 best.pt 基础上加强训练")
    parser.add_argument("--weights", type=str, default=None, help="指定原始预训练权重路径")
    parser.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    args = parser.parse_args()

    tasks = ["helmet", "plate", "action", "face"] if args.task == "all" else [args.task]
    for t in tasks:
        train_task(
            task=t,
            yolo_version=args.yolo,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            resume_from_best=args.resume_from_best,
            weights=args.weights,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
