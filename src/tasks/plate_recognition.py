"""
车牌识别 - YOLO 检测 + PaddleOCR 识别。

两步走：
1. YOLO 在图中找到车牌区域（bounding box）
2. 裁剪该区域，PaddleOCR 识别车牌文字
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import PROJECT_ROOT
from src.core.third_party_paths import PADDLEOCR_ROOT, bootstrap_env
from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper
from src.infer.inferencer import get_task_weights


class PlateRecognizer:
    """车牌检测 + 字符识别。"""

    def __init__(self, yolo_version: YOLOVersion = "yolov8", model_size: str = "n"):
        weights = get_task_weights("plate", yolo_version, model_size)
        self.detector = YOLOWrapper(version=yolo_version, weights=weights, model_size=model_size)
        self._ocr = None  # PaddleOCR 实例，首次使用时再创建（加载较慢）

    def _get_ocr(self):
        """懒加载 PaddleOCR，模型缓存目录: third_party/models/paddleocr/"""
        if self._ocr is None:
            try:
                bootstrap_env()
                from paddleocr import PaddleOCR

                self._ocr = PaddleOCR(use_angle_cls=True, lang="ch")
                print(f"[plate] PaddleOCR 缓存目录: {PADDLEOCR_ROOT}")
            except Exception as e:
                raise RuntimeError(f"PaddleOCR 初始化失败: {e}") from e
        return self._ocr

    def recognize(self, source: str | Path, conf: float = 0.25) -> list[dict[str, Any]]:
        """
        对单张图或目录下所有 jpg/png 做车牌识别。

        source 是目录时，glob 匹配常见图片后缀。
        """
        source = Path(source)
        if source.is_dir():
            files = list(source.glob("*.jpg")) + list(source.glob("*.png")) + list(source.glob("*.jpeg"))
        else:
            files = [source]

        outputs = []
        ocr = self._get_ocr()

        for img_path in files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            results = self.detector.predict(source=str(img_path), conf=conf, save=False)
            plates = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)  # 防止坐标越界
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

            # 检测器未检出时，对整图 OCR 作为兜底
            if not plates:
                text = self._ocr_plate(ocr, img)
                if text:
                    plates.append({"bbox": None, "confidence": 0.0, "plate_text": text, "fallback": True})

            outputs.append({"task": "plate", "path": str(img_path), "plates": plates})
        return outputs

    @staticmethod
    def _ocr_plate(ocr, crop: np.ndarray) -> str:
        """
        对裁剪块调用 OCR。

        PaddleOCR 返回嵌套结构；line[1][0] 为文字，line[1][1] 为置信度。
        """
        try:
            result = ocr.ocr(crop, cls=True)
            if not result or not result[0]:
                return ""
            texts = [line[1][0] for line in result[0] if line[1][1] > 0.5]
            return "".join(texts).replace(" ", "")
        except Exception:
            return ""
