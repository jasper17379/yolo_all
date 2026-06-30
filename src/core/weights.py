"""
YOLO 预训练权重与任务训练权重的统一路径解析。

目录约定:
  weights/pretrained/yolov8n.pt     # 官方预训练（download_pretrained_weights.py 下载）
  weights/{task}/best_yolov8n.pt    # 某任务 + 某规格训练得到的 best
  runs/train/{task}_yolov8n/        # Ultralytics 训练输出目录
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.core.config import PROJECT_ROOT, load_task_config

YOLOVersion = Literal["yolov5", "yolov8", "yolov10"]

YOLO_VERSIONS: tuple[YOLOVersion, ...] = ("yolov5", "yolov8", "yolov10")
MODEL_SIZES_COMMON = ("n", "s", "m", "l", "x")
MODEL_SIZES_V10_EXTRA = ("b",)  # YOLOv10 另有 b 规格

PRETRAINED_DIR = PROJECT_ROOT / "weights" / "pretrained"


def model_tag(yolo_version: YOLOVersion, model_size: str = "n") -> str:
    """组合权重标识，如 yolov8 + n → yolov8n。"""
    size = model_size.lower().strip()
    if size.startswith(yolo_version):
        return size
    return f"{yolo_version}{size}"


def best_weight_name(yolo_version: YOLOVersion, model_size: str = "n") -> str:
    """训练产出 best 文件名，如 best_yolov8n.pt。"""
    return f"best_{model_tag(yolo_version, model_size)}.pt"


def run_name(task: str, yolo_version: YOLOVersion, model_size: str = "n") -> str:
    """Ultralytics project name，如 plate_yolov8n。"""
    return f"{task}_{model_tag(yolo_version, model_size)}"


def list_pretrained_names() -> list[str]:
    """各版本常用预训练权重文件名列表。"""
    names: list[str] = []
    for ver in ("yolov5", "yolov8"):
        for s in MODEL_SIZES_COMMON:
            names.append(f"{ver}{s}.pt")
    for s in MODEL_SIZES_COMMON:
        names.append(f"yolov10{s}.pt")
    for s in MODEL_SIZES_V10_EXTRA:
        names.append(f"yolov10{s}.pt")
    return names


def resolve_pretrained(yolo_version: YOLOVersion, model_size: str = "n") -> Path:
    """
    解析预训练权重路径，优先级:
    1. weights/pretrained/{tag}.pt
    2. 仅文件名（交给 Ultralytics 自动下载）
    """
    tag = model_tag(yolo_version, model_size)
    local = PRETRAINED_DIR / f"{tag}.pt"
    if local.exists():
        return local
    return Path(f"{tag}.pt")


def task_weights_dir(task: str) -> Path:
    return PROJECT_ROOT / "weights" / task


def find_trained_best(task: str, yolo_version: YOLOVersion, model_size: str = "n") -> Path | None:
    """查找已训练 best 权重（按 yolo 版本 + 规格）。"""
    named = task_weights_dir(task) / best_weight_name(yolo_version, model_size)
    if named.exists():
        return named

    run_best = PROJECT_ROOT / "runs" / "train" / run_name(task, yolo_version, model_size) / "weights" / "best.pt"
    if run_best.exists():
        return run_best

    # 旧版 best.pt 仅在与默认 n 规格等价时回退，避免 s/m 误用旧 nano 权重
    if model_size == "n":
        legacy_best = task_weights_dir(task) / "best.pt"
        if legacy_best.exists():
            return legacy_best
        legacy_runs = sorted(
            (PROJECT_ROOT / "runs" / "train" / task).glob("*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if legacy_runs:
            return legacy_runs[0]

    return None


def get_task_weights(
    task: str,
    yolo_version: YOLOVersion = "yolov8",
    model_size: str = "n",
) -> Path:
    """
    推理/训练时解析任务权重，优先级:
    1. weights/{task}/best_{tag}.pt
    2. runs/train/{task}_{tag}/weights/best.pt
    3. runs/train/{task}/*/weights/best.pt（旧目录，兼容）
    4. weights/{task}/best.pt（旧命名，兼容）
    5. weights/pretrained/{tag}.pt 或 Ultralytics 自动下载
    """
    trained = find_trained_best(task, yolo_version, model_size)
    if trained is not None:
        return trained

    cfg = load_task_config(task)
    cfg_default = cfg.get("default_weights")
    if cfg_default and model_size == "n" and yolo_version == "yolov8":
        cfg_path = Path(cfg_default)
        if cfg_path.is_absolute() and cfg_path.exists():
            return cfg_path
        local_cfg = PRETRAINED_DIR / Path(cfg_default).name
        if local_cfg.exists():
            return local_cfg

    return resolve_pretrained(yolo_version, model_size)


def sync_trained_best(task: str, yolo_version: YOLOVersion, model_size: str, best_src: str | Path) -> Path:
    """训练完成后把 best.pt 同步到 weights/{task}/best_{tag}.pt。"""
    dest = task_weights_dir(task) / best_weight_name(yolo_version, model_size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(best_src, dest)
    return dest
