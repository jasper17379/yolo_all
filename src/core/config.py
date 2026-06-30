"""
项目路径与配置工具。

负责：
- 定位项目根目录 PROJECT_ROOT
- 读取 YAML 配置文件（global / 各任务 task）
- 创建数据集、权重等目录
"""

from __future__ import annotations  # 见 demo.py 说明

from pathlib import Path  # 跨平台路径操作
from typing import Any  # Any 表示「任意类型」，用于 YAML 解析后的 dict 值

import yaml  # PyYAML：读写 .yaml 配置文件

# __file__ 是当前文件 config.py 的路径
# parents[0]=core, parents[1]=src, parents[2]=项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    读取 YAML 文件并返回 Python 字典。

    str | Path：类型注解，表示参数可以是字符串或 Path 对象（Python 3.10+ 写法）
    with open(...) as f：上下文管理器，退出 with 块时自动关闭文件
    yaml.safe_load：比 load 更安全，不会执行 YAML 里的任意 Python 对象
    or {}：若文件为空，safe_load 返回 None，则改用空字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_global_config() -> dict[str, Any]:
    """读取 configs/global.yaml（全局默认参数，如默认权重名）。"""
    return load_yaml(PROJECT_ROOT / "configs" / "global.yaml")


def load_task_config(task: str) -> dict[str, Any]:
    """
    读取单个任务的配置，例如 task='helmet' → configs/tasks/helmet.yaml。

    配置里通常包含：dataset 路径、classes 类别、default_weights 等。
    """
    path = PROJECT_ROOT / "configs" / "tasks" / f"{task}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"任务配置不存在: {path}")
    return load_yaml(path)


def resolve_path(relative: str | Path) -> Path:
    """
    把相对路径转成基于项目根的绝对 Path。

    若传入的已是绝对路径，则原样返回。
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_dirs() -> None:
    """
    创建项目运行所需的目录结构（若已存在则跳过）。

    mkdir(parents=True, exist_ok=True)：
    - parents=True：中间目录不存在则一并创建
    - exist_ok=True：目录已存在不报错
    """
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
        "weights/pretrained",
        "runs",
        "outputs",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
