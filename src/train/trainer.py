"""
统一训练入口。

示例:
  python -m src.train.trainer --task helmet --device auto --epochs 20
  python -m src.train.trainer --task plate --device 0 --batch 16 --lr0 0.001
  python -m src.train.trainer --task helmet --device 0,1 --epochs 50
  python -m src.train.trainer --task face --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import PROJECT_ROOT, ensure_dirs, load_global_config, load_task_config
from src.core.dataset_yaml import prepare_data_yaml
from src.core.device import add_device_arg, device_label, print_device_info, resolve_yolo_device
from src.core.third_party_paths import bootstrap_env
from src.core.train_config import TrainHyperParams
from src.core.weights import find_trained_best, resolve_pretrained, run_name, sync_trained_best
from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper


def train_task(
    task: str,
    yolo_version: YOLOVersion = "yolov8",
    model_size: str = "n",
    device: str = "auto",
    hyper: TrainHyperParams | None = None,
    resume_from_best: bool = False,
    weights: str | None = None,
    resume: bool = False,
) -> dict:
    ensure_dirs()
    bootstrap_env()
    print_device_info(device)

    task_cfg = load_task_config(task)
    hyper = hyper or TrainHyperParams.from_global()

    if task_cfg.get("type") == "recognition":
        from src.tasks.face_trainer import train_face_gallery

        return train_face_gallery(
            max_images_per_person=task_cfg.get("max_images_per_person", 20),
            device=device,
        )

    data_yaml = prepare_data_yaml(task_cfg["dataset"])
    yolo_device = resolve_yolo_device(device)

    if weights:
        init_weights: str | Path = Path(weights) if Path(weights).exists() else weights
    elif resume_from_best:
        best_path = find_trained_best(task, yolo_version, model_size)
        if best_path:
            init_weights = best_path
            print(f"[{task}] 基于已训练 best 继续: {best_path}")
        else:
            init_weights = resolve_pretrained(yolo_version, model_size)
            print(f"[{task}] 未找到 best_{yolo_version}{model_size}，改用预训练: {init_weights}")
    else:
        init_weights = resolve_pretrained(yolo_version, model_size)
        print(f"[{task}] 从预训练权重开始: {init_weights}")

    wrapper = YOLOWrapper(version=yolo_version, weights=str(init_weights), model_size=model_size)
    exp_name = run_name(task, yolo_version, model_size)

    train_kw = hyper.to_ultralytics_kwargs(yolo_device)
    print(
        f"[{task}] 训练超参: epochs={hyper.epochs} batch={hyper.batch} imgsz={hyper.imgsz} "
        f"lr0={hyper.lr0} patience={hyper.patience} amp={hyper.amp} device={device_label(device)}"
    )

    result = wrapper.train(
        data_yaml=data_yaml,
        epochs=hyper.epochs,
        batch=hyper.batch,
        imgsz=hyper.imgsz,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=exp_name,
        resume=resume,
        **train_kw,
    )

    dest = sync_trained_best(task, yolo_version, model_size, result["best"])
    print(f"[{task}] 训练完成, best 权重: {dest}")
    print(f"[{task}] 原始 run 目录: {result['save_dir']}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    g = load_global_config()
    t = g.get("train", {})
    p = argparse.ArgumentParser(description="Vision AI 统一训练")
    p.add_argument("--task", required=True, choices=["helmet", "plate", "action", "face", "all"])
    p.add_argument("--yolo", default=g.get("default_yolo_version", "yolov8"), choices=["yolov5", "yolov8", "yolov10"])
    p.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x", "b"])
    add_device_arg(p, default=g.get("device", "auto"))

    p.add_argument("--epochs", type=int, default=t.get("epochs", 20))
    p.add_argument("--batch", type=int, default=t.get("batch", 8))
    p.add_argument("--imgsz", type=int, default=t.get("imgsz", 640))
    p.add_argument("--patience", type=int, default=t.get("patience", 50))
    p.add_argument("--workers", type=int, default=t.get("workers", 4))
    p.add_argument("--lr0", type=float, default=t.get("lr0", 0.01), help="初始学习率")
    p.add_argument("--lrf", type=float, default=t.get("lrf", 0.01), help="最终学习率因子")
    p.add_argument("--optimizer", default=t.get("optimizer", "auto"), choices=["auto", "SGD", "Adam", "AdamW"])
    p.add_argument("--momentum", type=float, default=t.get("momentum", 0.937))
    p.add_argument("--weight-decay", type=float, default=t.get("weight_decay", 0.0005))
    p.add_argument("--warmup-epochs", type=float, default=t.get("warmup_epochs", 3.0))
    p.add_argument("--mosaic", type=float, default=t.get("mosaic", 1.0), help="马赛克增强概率 0~1")
    p.add_argument("--cos-lr", action="store_true", default=bool(t.get("cos_lr", False)))
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度(AMP)")
    p.add_argument("--freeze", type=int, default=t.get("freeze"), help="冻结前 N 层")
    p.add_argument("--seed", type=int, default=t.get("seed", 0))
    p.add_argument("--close-mosaic", type=int, default=t.get("close_mosaic", 10))

    p.add_argument("--resume-from-best", action="store_true")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    hyper = TrainHyperParams.from_global(
        {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "patience": args.patience,
            "workers": args.workers,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "mosaic": args.mosaic,
            "cos_lr": args.cos_lr,
            "amp": not args.no_amp,
            "freeze": args.freeze,
            "seed": args.seed,
            "close_mosaic": args.close_mosaic,
        }
    )

    tasks = ["helmet", "plate", "action", "face"] if args.task == "all" else [args.task]
    for t in tasks:
        train_task(
            task=t,
            yolo_version=args.yolo,
            model_size=args.model_size,
            device=args.device,
            hyper=hyper,
            resume_from_best=args.resume_from_best,
            weights=args.weights,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
