"""训练超参：从 global.yaml 读取默认值并合并 CLI 参数。"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from src.core.config import load_global_config


@dataclass
class TrainHyperParams:
    epochs: int = 20
    batch: int = 8
    imgsz: int = 640
    patience: int = 50
    workers: int = 4
    lr0: float = 0.01
    lrf: float = 0.01
    optimizer: str = "auto"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    mosaic: float = 1.0
    cos_lr: bool = False
    amp: bool = True
    freeze: int | None = None
    seed: int = 0
    close_mosaic: int = 10

    @classmethod
    def from_global(cls, overrides: dict[str, Any] | None = None) -> "TrainHyperParams":
        cfg = load_global_config().get("train", {})
        kwargs = {f.name: cfg.get(f.name, getattr(cls, f.name)) for f in fields(cls)}
        if overrides:
            for k, v in overrides.items():
                if v is not None and k in kwargs:
                    kwargs[k] = v
        return cls(**kwargs)

    def to_ultralytics_kwargs(self, device: str | int) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "device": device,
            "patience": self.patience,
            "workers": self.workers,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "mosaic": self.mosaic,
            "cos_lr": self.cos_lr,
            "amp": self.amp,
            "seed": self.seed,
            "close_mosaic": self.close_mosaic,
        }
        if self.freeze is not None:
            kw["freeze"] = self.freeze
        return kw


@dataclass
class InferHyperParams:
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    half: bool = False

    @classmethod
    def from_global(cls, overrides: dict[str, Any] | None = None) -> "InferHyperParams":
        cfg = load_global_config().get("infer", {})
        kwargs = {f.name: cfg.get(f.name, getattr(cls, f.name)) for f in fields(cls)}
        if overrides:
            for k, v in overrides.items():
                if v is not None and k in kwargs:
                    kwargs[k] = v
        return cls(**kwargs)
