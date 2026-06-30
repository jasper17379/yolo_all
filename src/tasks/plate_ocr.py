"""
车牌字符识别引擎（预训练，无需训练）。

源码与模型:
  - HyperLPR3: third_party/HyperLPR/Prj-Python + third_party/models/hyperlpr3/
  - PaddleOCR: pip + third_party/models/paddleocr/ (PADDLE_PDX_CACHE_HOME)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.device import paddle_use_gpu
from src.core.third_party_paths import (
    bootstrap_env,
    hyperlpr3_available,
    prepare_hyperlpr3,
)


@dataclass
class PlateOCRResult:
    text: str
    confidence: float
    plate_type: str = ""
    bbox: list[int] | None = None


PLATE_TYPE_NAMES = {
    0: "蓝牌",
    1: "黄牌单层",
    2: "白牌",
    3: "绿牌",
    4: "黑牌",
    5: "港牌",
    6: "澳牌",
    7: "双层黄牌",
}


def normalize_plate_text(text: str) -> str:
    return (
        text.replace(" ", "")
        .replace("·", "")
        .replace("-", "")
        .replace("O", "0")
        .strip()
        .upper()
    )


class HyperLPR3Engine:
    """HyperLPR3 — 使用 third_party/HyperLPR 源码。"""

    def __init__(self, device: str = "auto"):
        self.device = device
        self._catcher = None

    def _get_catcher(self):
        if self._catcher is None:
            if not prepare_hyperlpr3():
                raise RuntimeError(
                    "HyperLPR3 不可用。请运行: python scripts/setup_third_party.py --download-hyperlpr-models"
                )
            import hyperlpr3 as lpr3

            try:
                self._catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
            except TypeError:
                self._catcher = lpr3.LicensePlateCatcher()
        return self._catcher

    def recognize(self, image: np.ndarray) -> list[PlateOCRResult]:
        catcher = self._get_catcher()
        try:
            raw = catcher(image)
        except Exception:
            return []
        results: list[PlateOCRResult] = []
        for item in raw or []:
            if not item or len(item) < 2:
                continue
            code, conf = str(item[0]), float(item[1])
            ptype = PLATE_TYPE_NAMES.get(int(item[2]), "") if len(item) > 2 else ""
            box = None
            if len(item) > 3 and item[3] is not None:
                b = item[3]
                if len(b) >= 4:
                    box = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
            text = normalize_plate_text(code)
            if text:
                results.append(PlateOCRResult(text=text, confidence=conf, plate_type=ptype, bbox=box))
        return results


class PaddleOCREngine:
    """PaddleOCR 通用中文 OCR（回退）。"""

    def __init__(self, device: str = "auto"):
        self.device = device
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from src.core.third_party_paths import PADDLEOCR_ROOT

            bootstrap_env()
            from paddleocr import PaddleOCR

            use_gpu = paddle_use_gpu(self.device)
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", device="gpu:0" if use_gpu else "cpu")
            except TypeError:
                self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu)
            print(f"[plate-ocr] PaddleOCR 缓存: {PADDLEOCR_ROOT} | gpu={use_gpu}")
        return self._ocr

    def recognize(self, image: np.ndarray) -> list[PlateOCRResult]:
        ocr = self._get_ocr()
        try:
            raw = ocr.ocr(image, cls=True)
        except Exception:
            return []
        if not raw or not raw[0]:
            return []
        parts: list[tuple[str, float]] = []
        for line in raw[0]:
            if line[1][1] > 0.5:
                parts.append((line[1][0], float(line[1][1])))
        if not parts:
            return []
        text = normalize_plate_text("".join(p[0] for p in parts))
        conf = sum(p[1] for p in parts) / len(parts)
        return [PlateOCRResult(text=text, confidence=conf)] if text else []


def create_plate_ocr_engine(engine: str = "auto", device: str = "auto") -> HyperLPR3Engine | PaddleOCREngine:
    bootstrap_env()
    if engine == "paddleocr":
        return PaddleOCREngine(device)
    if engine == "hyperlpr3":
        return HyperLPR3Engine(device)
    if hyperlpr3_available():
        return HyperLPR3Engine(device)
    print("[plate-ocr] HyperLPR3 未就绪，回退 PaddleOCR（运行 setup_third_party.py --download-hyperlpr-models）")
    return PaddleOCREngine(device)


def ocr_on_crop(
    engine_name: str,
    device: str,
    crop: np.ndarray,
    engines: dict[str, Any] | None = None,
) -> PlateOCRResult | None:
    engines = engines or {}
    order = ["hyperlpr3", "paddleocr"] if engine_name == "auto" else [engine_name]
    for name in order:
        if name == "hyperlpr3" and not hyperlpr3_available():
            continue
        if name not in engines:
            try:
                engines[name] = create_plate_ocr_engine(name, device)
            except Exception as e:
                print(f"[plate-ocr] {name} 初始化失败: {e}")
                continue
        eng = engines[name]
        try:
            hits = eng.recognize(crop)
            if hits:
                best = max(hits, key=lambda x: x.confidence)
                if best.text:
                    return best
        except Exception as e:
            print(f"[plate-ocr] {name} 识别失败: {e}")
    return None
