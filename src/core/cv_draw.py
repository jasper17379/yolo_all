"""OpenCV 图像上绘制中文标签（cv2.putText 不支持中文会显示 ??）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in candidates:
        if Path(fp).exists():
            _FONT_CACHE[size] = ImageFont.truetype(fp, size)
            return _FONT_CACHE[size]
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def text_size(text: str, font_size: int = 18) -> tuple[int, int]:
    font = _get_font(font_size)
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def put_text_bgr(
    bgr: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 18,
) -> np.ndarray:
    """在 BGR 图像上绘制文本（支持中文）。"""
    if not text:
        return bgr
    pil = _bgr_to_pil(bgr)
    draw = ImageDraw.Draw(pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(xy, text, font=_get_font(font_size), fill=color_rgb)
    return _pil_to_bgr(pil)


def draw_label_bgr(
    bgr: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    box_color_bgr: tuple[int, int, int] = (0, 0, 255),
    text_color_bgr: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 18,
    padding: int = 4,
) -> np.ndarray:
    """
    在 anchor 上方绘制带背景色的标签（用于检测框上方）。

    anchor: 框左上角 (x1, y1)
    """
    if not text:
        return bgr
    tw, th = text_size(text, font_size)
    x1, y1 = anchor
    bar_x2 = x1 + tw + padding * 2
    bar_y1 = max(0, y1 - th - padding * 2)
    bar_y2 = y1
    out = bgr.copy()
    cv2.rectangle(out, (x1, bar_y1), (bar_x2, bar_y2), box_color_bgr, -1)
    return put_text_bgr(out, text, (x1 + padding, bar_y1 + padding // 2), text_color_bgr, font_size)
