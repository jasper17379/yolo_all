"""人脸库构建 - 从 LFW 等数据集初始化 gallery"""

from __future__ import annotations

from pathlib import Path

import cv2

from src.core.config import PROJECT_ROOT, resolve_path
from src.tasks.face_recognition import FaceRecognizer


def train_face_gallery(max_persons: int | None = None) -> dict:
    """从 datasets/face/lfw 构建初始人脸库。max_persons=None 时录入全部。"""
    lfw_dir = resolve_path("datasets/face/lfw")
    rec = FaceRecognizer()
    enrolled = 0
    failed = 0

    if not lfw_dir.exists():
        return {"task": "face", "message": "LFW 数据集未下载，请先运行 download_datasets.py", "enrolled": 0}

    persons = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    if max_persons is not None:
        persons = persons[:max_persons]

    for person_dir in persons:
        result = rec.enroll_from_dir(person_dir)
        if result["success"]:
            enrolled += 1
        else:
            failed += 1

    return {
        "task": "face",
        "enrolled": enrolled,
        "failed": failed,
        "total_images": sum(
            len(list((rec.gallery_dir / n).glob("*")))
            for n in rec.list_persons()
            if (rec.gallery_dir / n).exists()
        ),
        "gallery": str(rec.gallery_dir),
        "backend": rec._backend if rec._app else "pending",
    }


def build_gallery_from_custom(custom_dir: str | Path) -> dict:
    """从自定义目录构建人脸库，目录结构: custom_dir/person_name/*.jpg"""
    custom_dir = Path(custom_dir)
    rec = FaceRecognizer()
    count = 0
    for person_dir in custom_dir.iterdir():
        if person_dir.is_dir():
            r = rec.enroll_from_dir(person_dir)
            count += r.get("count", 0)
    return {"enrolled_persons": len(rec.list_persons()), "images": count}
