"""
车牌识别 - YOLO 检测定位 + HyperLPR3/PaddleOCR 读字。

流程:
  1. YOLO 在场景图中框出车牌区域（可训练）
  2. 裁剪车牌区域 → OCR 识别车牌号（预训练模型，无需标注训练）
  3. YOLO 未检出时，可回退到 HyperLPR3 端到端检测+识别
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import load_global_config, load_task_config
from src.core.device import resolve_yolo_device
from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper
from src.core.weights import get_task_weights
from src.tasks.plate_ocr import HyperLPR3Engine, PlateOCRResult, create_plate_ocr_engine, ocr_on_crop


class PlateRecognizer:
    """车牌检测 + 字符识别。"""

    def __init__(
        self,
        yolo_version: YOLOVersion = "yolov8",
        model_size: str = "n",
        device: str | None = None,
        ocr_engine: str | None = None,
        fallback_e2e: bool = True,
    ):
        task_cfg = load_task_config("plate")
        g = load_global_config()
        self.device = device or g.get("device", "auto")
        self.ocr_engine = ocr_engine or task_cfg.get("ocr_engine", "auto")
        self.fallback_e2e = fallback_e2e
        weights = get_task_weights("plate", yolo_version, model_size)
        self.detector = YOLOWrapper(version=yolo_version, weights=weights, model_size=model_size)
        self._ocr_engines: dict[str, Any] = {}
        self._e2e: HyperLPR3Engine | None = None

    def _e2e_recognize(self, img: np.ndarray) -> list[PlateOCRResult]:
        """HyperLPR3 端到端：自带检测+识别，YOLO 失败时使用。"""
        if self.ocr_engine == "paddleocr":
            return []
        try:
            if self._e2e is None:
                eng = create_plate_ocr_engine(
                    "hyperlpr3" if self.ocr_engine == "hyperlpr3" else "auto",
                    self.device,
                )
                if not isinstance(eng, HyperLPR3Engine):
                    return []
                self._e2e = eng
            return self._e2e.recognize(img)
        except Exception as e:
            print(f"[plate] 端到端识别跳过: {e}")
            return []

    def recognize_frame(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> list[dict[str, Any]]:
        """对内存中的 BGR 帧做检测+识别，供 demo 实时调用。"""
        yolo_device = resolve_yolo_device(self.device)
        results = self.detector.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=yolo_device,
            save=False,
            verbose=False,
        )
        plates: list[dict[str, Any]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ocr = ocr_on_crop(self.ocr_engine, self.device, crop, self._ocr_engines)
                plates.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(box.conf[0]),
                        "plate_text": ocr.text if ocr else "",
                        "ocr_confidence": ocr.confidence if ocr else 0.0,
                        "plate_type": ocr.plate_type if ocr else "",
                    }
                )

        if not plates and self.fallback_e2e:
            for hit in self._e2e_recognize(frame):
                plates.append(
                    {
                        "bbox": hit.bbox,
                        "confidence": hit.confidence,
                        "plate_text": hit.text,
                        "ocr_confidence": hit.confidence,
                        "plate_type": hit.plate_type,
                        "fallback": True,
                    }
                )
        return plates

    def recognize(
        self,
        source: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> list[dict[str, Any]]:
        source = Path(source)
        if source.is_dir():
            files = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png")) + sorted(source.glob("*.jpeg"))
        else:
            files = [source]

        yolo_device = resolve_yolo_device(self.device)
        outputs = []

        for img_path in files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            plates = self.recognize_frame(img, conf=conf, imgsz=imgsz)
            outputs.append(
                {
                    "task": "plate",
                    "path": str(img_path),
                    "device": str(yolo_device),
                    "ocr_engine": self.ocr_engine,
                    "plates": plates,
                }
            )
        return outputs
