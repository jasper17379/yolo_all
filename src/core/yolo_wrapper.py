"""
统一 YOLO 模型封装 - 支持 YOLOv5 / YOLOv8 / YOLOv10。

底层使用 `third_party/ultralytics` 源码（`bootstrap_env` 自动加载），本类统一 train / predict / export 接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal  # Literal：限定变量只能是几个字符串之一

# YOLOVersion 是类型别名，只能是这三个字符串之一，IDE 和类型检查器会提示
YOLOVersion = Literal["yolov5", "yolov8", "yolov10"]


class YOLOWrapper:
    """统一训练/推理接口，底层使用 Ultralytics。"""

    SUPPORTED = ("yolov5", "yolov8", "yolov10")  # 类属性，所有实例共享

    def __init__(
        self,
        version: YOLOVersion = "yolov8",
        weights: str | Path | None = None,
        model_size: str = "n",
    ):
        """
        参数:
            version: YOLO 版本
            weights: .pt 权重文件路径；None 时使用 weights/pretrained/{version}{size}.pt
            model_size: 模型规模 n/s/m/l/x（与 weights 二选一，weights 优先）
        """
        if version not in self.SUPPORTED:
            raise ValueError(f"不支持的 YOLO 版本: {version}, 可选: {self.SUPPORTED}")
        self.version = version
        self.model_size = model_size
        if weights is not None:
            self.weights = str(weights)
        else:
            from src.core.weights import resolve_pretrained

            self.weights = str(resolve_pretrained(version, model_size))
        self._model = None  # 延迟加载：构造时不立刻加载大模型，首次 predict 再 load

    def load(self):
        """从 third_party/ultralytics 加载 YOLO（优先 vendored，非 pip）。"""
        from src.core.third_party_paths import import_yolo

        YOLO = import_yolo()
        self._model = YOLO(self.weights)
        return self

    @property
    def model(self):
        """
        @property：把方法当属性用，写 wrapper.model 实际调用此方法。

        若尚未 load，则自动 load。
        """
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
        """
        训练模型。

        data_yaml: YOLO 格式的数据集描述文件（train/val 路径、类别数等）
        **kwargs: 接收额外关键字参数并传给 Ultralytics（如 device='0'）
        """
        args: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "project": project,
            "name": name,
            "exist_ok": True,  # 同名实验目录已存在时覆盖/继续
            **kwargs,
        }
        if resume:
            args["resume"] = True  # 从 last.pt 断点续训
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
        """
        推理。

        source 可以是：图片路径、目录、视频、numpy 数组（BGR）、摄像头 id 等。
        conf: 置信度阈值；iou: NMS 重叠阈值。
        """
        # 已是 numpy 数组时不能 str()，否则会变成无意义的字符串
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
        """导出模型用于 C++ / RK3588 等部署（onnx、engine 等格式）。"""
        path = self.model.export(format=fmt, **kwargs)
        return str(path)

    @classmethod
    def from_best_or_pretrained(
        cls,
        version: YOLOVersion,
        best_path: str | Path | None,
        pretrained: str | Path | None = None,
    ) -> "YOLOWrapper":
        """
        @classmethod：第一个参数是类本身 cls，用于工厂方法创建实例。

        优先用已训练好的 best.pt；否则用指定预训练；都没有则用默认 nano。
        """
        if best_path and Path(best_path).exists():
            return cls(version=version, weights=best_path)
        if pretrained:
            return cls(version=version, weights=pretrained)
        return cls(version=version)
