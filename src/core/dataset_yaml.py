"""YOLO data.yaml 路径校正，避免旧项目绝对路径导致训练失败。"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core.config import load_yaml, resolve_path


def prepare_data_yaml(dataset_yaml: str | Path) -> Path:
    """
    将 data.yaml 的 path 重写为当前数据集目录的绝对路径，并校验 train/val 图像存在。

    返回写入的 data.resolved.yaml 路径（传给 Ultralytics）。
    """
    yaml_path = resolve_path(dataset_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"数据集配置不存在: {yaml_path}")

    data = load_yaml(yaml_path)
    base = yaml_path.parent.resolve()
    data["path"] = str(base)

    for key in ("train", "val"):
        rel = data.get(key, f"images/{key}")
        img_dir = (base / rel).resolve()
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"数据集 [{base.name}] 缺少 {key} 图像目录: {img_dir}\n"
                f"请先运行: python scripts/import_external_datasets.py 或 python scripts/download_datasets.py"
            )
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        if not images:
            raise FileNotFoundError(f"数据集 [{base.name}] {key} 目录为空: {img_dir}")

    out = base / "data.resolved.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return out
