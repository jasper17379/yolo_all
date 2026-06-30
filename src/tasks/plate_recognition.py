"""
车牌识别 - YOLO 检测 + PaddleOCR 识别。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import load_global_config
from src.core.device import paddle_use_gpu, resolve_yolo_device
from src.core.third_party_paths import PADDLEOCR_ROOT, bootstrap_env
from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper
from src.core.weights import get_task_weights


class PlateRecognizer:
    """车牌检测 + 字符识别。"""

    def __init__(
        self,
        yolo_version: YOLOVersion = "yolov8",
        model_size: str = "n",
        device: str | None = None,
    ):
        self.device = device or load_global_config().get("device", "auto")
        weights = get_task_weights("plate", yolo_version, model_size)
        self.detector = YOLOWrapper(version=yolo_version, weights=weights, model_size=model_size)
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            try:
                bootstrap_env()
                from paddleocr import PaddleOCR

                use_gpu = paddle_use_gpu(self.device)
                try:
                    self._ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang="ch",
                        device="gpu:0" if use_gpu else "cpu",
                    )
                except TypeError:
                    self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu)
                print(f"[plate] PaddleOCR 缓存: {PADDLEOCR_ROOT} | gpu={use_gpu}")
            except Exception as e:
                raise RuntimeError(f"PaddleOCR 初始化失败: {e}") from e
        return self._ocr

    def recognize(
        self,
        source: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> list[dict[str, Any]]:
        source = Path(source)
        if source.is_dir():
            files = list(source.glob("*.jpg")) + list(source.glob("*.png")) + list(source.glob("*.jpeg"))
        else:
            files = [source]

        yolo_device = resolve_yolo_device(self.device)
        outputs = []
        ocr = self._get_ocr()

        for img_path in files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            results = self.detector.predict(
                source=str(img_path),
                conf=conf,
                imgsz=imgsz,
                device=yolo_device,
                save=False,
            )
            plates = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    text = self._ocr_plate(ocr, crop)
                    plates.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(box.conf[0]),
                            "plate_text": text,
                        }
                    )

            if not plates:
                text = self._ocr_plate(ocr, img)
                if text:
                    plates.append({"bbox": None, "confidence": 0.0, "plate_text": text, "fallback": True})

            outputs.append({"task": "plate", "path": str(img_path), "device": str(yolo_device), "plates": plates})
        return outputs

    @staticmethod
    def _ocr_plate(ocr, crop: np.ndarray) -> str:
        try:
            result = ocr.ocr(crop, cls=True)
            if not result or not result[0]:
                return ""
            texts = [line[1][0] for line in result[0] if line[1][1] > 0.5]
            return "".join(texts).replace(" ", "")
        except Exception:
            return ""
