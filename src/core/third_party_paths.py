"""
third_party 目录统一路径：源码仓库 + 业务模型权重。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.core.config import PROJECT_ROOT

THIRD_PARTY = PROJECT_ROOT / "third_party"
MODELS_ROOT = THIRD_PARTY / "models"
INSIGHTFACE_ROOT = MODELS_ROOT / "insightface"
PADDLEOCR_ROOT = MODELS_ROOT / "paddleocr"
WEIGHTS_PRETRAINED = PROJECT_ROOT / "weights" / "pretrained"


def ensure_third_party_dirs() -> None:
    for d in (
        THIRD_PARTY,
        MODELS_ROOT,
        INSIGHTFACE_ROOT,
        INSIGHTFACE_ROOT / "models",
        PADDLEOCR_ROOT,
        THIRD_PARTY / "ultralytics_config",
        WEIGHTS_PRETRAINED,
    ):
        d.mkdir(parents=True, exist_ok=True)


def insightface_model_dir(model_name: str = "buffalo_l") -> Path:
    return INSIGHTFACE_ROOT / "models" / model_name


def migrate_insightface_from_user_home(model_name: str = "buffalo_l") -> Path | None:
    src = Path.home() / ".insightface" / "models" / model_name
    dst = insightface_model_dir(model_name)
    if not src.exists() or not any(src.iterdir()):
        return None
    if dst.exists() and any(dst.iterdir()):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def bootstrap_env() -> None:
    ensure_third_party_dirs()
    migrate_insightface_from_user_home()
    os.environ.setdefault("INSIGHTFACE_HOME", str(INSIGHTFACE_ROOT))
    os.environ.setdefault("PADDLE_PDX_HOME", str(PADDLEOCR_ROOT))
    os.environ.setdefault("PADDLEX_HOME", str(PADDLEOCR_ROOT))
    os.environ.setdefault("PADDLEOCR_HOME", str(PADDLEOCR_ROOT))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "third_party" / "ultralytics_config"))
