"""项目路径与配置工具"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_global_config() -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "configs" / "global.yaml")


def load_task_config(task: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "tasks" / f"{task}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"任务配置不存在: {path}")
    return load_yaml(path)


def resolve_path(relative: str | Path) -> Path:
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_dirs() -> None:
    dirs = [
        "datasets/face/gallery",
        "datasets/face/raw",
        "datasets/face/lfw",
        "datasets/plate/images/train",
        "datasets/plate/images/val",
        "datasets/plate/labels/train",
        "datasets/plate/labels/val",
        "datasets/helmet/images/train",
        "datasets/helmet/images/val",
        "datasets/helmet/labels/train",
        "datasets/helmet/labels/val",
        "datasets/action/images/train",
        "datasets/action/images/val",
        "datasets/action/labels/train",
        "datasets/action/labels/val",
        "datasets/custom/images",
        "datasets/custom/labels",
        "weights",
        "runs",
        "outputs",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
