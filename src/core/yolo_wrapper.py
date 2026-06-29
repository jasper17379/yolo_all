"""统一 YOLO 模型封装 - 支持 YOLOv5 / YOLOv8 / YOLOv10"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

YOLOVersion = Literal["yolov5", "yolov8", "yolov10"]


class YOLOWrapper:
    """统一训练/推理接口，底层使用 Ultralytics。"""

    SUPPORTED = ("yolov5", "yolov8", "yolov10")

    def __init__(self, version: YOLOVersion = "yolov8", weights: str | Path | None = None):
        if version not in self.SUPPORTED:
            raise ValueError(f"不支持的 YOLO 版本: {version}, 可选: {self.SUPPORTED}")
        self.version = version
        self.weights = str(weights) if weights else self._default_weights(version)
        self._model = None

    @staticmethod
    def _default_weights(version: YOLOVersion) -> str:
        mapping = {
            "yolov5": "yolov5n.pt",
            "yolov8": "yolov8n.pt",
            "yolov10": "yolov10n.pt",
        }
        return mapping[version]

    def load(self):
        from ultralytics import YOLO

        self._model = YOLO(self.weights)
        return self

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def train(
        self,
        data_yaml: str | Path,
        epochs: int = 20,
        batch: int = 8,
        imgsz: int = 640,
        project: str = "runs/train",
        name: str = "exp",
        resume: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """训练模型。resume=True 时从 last.pt 继续；否则从 self.weights 开始。"""
        args: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "project": project,
            "name": name,
            "exist_ok": True,
            **kwargs,
        }
        if resume:
            args["resume"] = True
        results = self.model.train(**args)
        save_dir = Path(self.model.trainer.save_dir)
        return {
            "save_dir": str(save_dir),
            "best": str(save_dir / "weights" / "best.pt"),
            "last": str(save_dir / "weights" / "last.pt"),
            "results": results,
        }

    def predict(
        self,
        source: str | Path | Any,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = True,
        project: str = "runs/predict",
        name: str = "exp",
        **kwargs: Any,
    ) -> list[Any]:
        src = source if not isinstance(source, (str, Path)) else str(source)
        return self.model.predict(
            source=src,
            conf=conf,
            iou=iou,
            save=save,
            project=project,
            name=name,
            exist_ok=True,
            **kwargs,
        )

    def export(self, fmt: str = "onnx", **kwargs: Any) -> str:
        """导出模型用于 C++ / RK3588 部署。"""
        path = self.model.export(format=fmt, **kwargs)
        return str(path)

    @classmethod
    def from_best_or_pretrained(
        cls,
        version: YOLOVersion,
        best_path: str | Path | None,
        pretrained: str | Path | None = None,
    ) -> "YOLOWrapper":
        """优先使用 best 权重继续训练，否则使用指定预训练权重。"""
        if best_path and Path(best_path).exists():
            return cls(version=version, weights=best_path)
        if pretrained:
            return cls(version=version, weights=pretrained)
        return cls(version=version)
