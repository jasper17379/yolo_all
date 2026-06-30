"""人脸库构建 - 从 LFW 等数据集初始化 gallery。"""

from __future__ import annotations

from pathlib import Path

from src.core.config import resolve_path
from src.core.device import print_device_info
from src.core.third_party_paths import bootstrap_env, insightface_model_dir
from src.tasks.face_recognition import FaceRecognizer


def train_face_gallery(
    max_persons: int | None = None,
    max_images_per_person: int = 20,
    device: str = "auto",
) -> dict:
    bootstrap_env()
    print_device_info(device)
    lfw_dir = resolve_path("datasets/face/lfw")
    if not lfw_dir.exists():
        return {
            "task": "face",
            "message": "LFW 数据集未下载，请先运行: python scripts/import_external_datasets.py 或 download_datasets.py",
            "enrolled": 0,
        }

    persons = sorted([d for d in lfw_dir.iterdir() if d.is_dir()])
    if not persons:
        return {"task": "face", "message": f"未找到人脸子目录: {lfw_dir}", "enrolled": 0}
    if max_persons is not None:
        persons = persons[:max_persons]

    rec = FaceRecognizer(device=device)
    enrolled = 0
    failed = 0
    total = len(persons)

    for i, person_dir in enumerate(persons, 1):
        print(f"[face] 录入 {i}/{total}: {person_dir.name}")
        result = rec.enroll_from_dir(person_dir, max_images=max_images_per_person)
        if result["success"]:
            enrolled += 1
            print(f"  -> ok ({result.get('count', 0)} 张)")
        else:
            failed += 1
            print(f"  -> 失败")

    return {
        "task": "face",
        "device": device,
        "enrolled": enrolled,
        "failed": failed,
        "total_images": sum(
            len(list((rec.gallery_dir / n).glob("*")))
            for n in rec.list_persons()
            if (rec.gallery_dir / n).exists()
        ),
        "gallery": str(rec.gallery_dir),
        "backend": rec._backend if rec._app else "pending",
        "model_dir": str(insightface_model_dir(rec.model_name)),
    }


def build_gallery_from_custom(custom_dir: str | Path, device: str = "auto") -> dict:
    custom_dir = Path(custom_dir)
    bootstrap_env()
    rec = FaceRecognizer(device=device)
    count = 0
    for person_dir in custom_dir.iterdir():
        if person_dir.is_dir():
            r = rec.enroll_from_dir(person_dir)
            count += r.get("count", 0)
    return {"enrolled_persons": len(rec.list_persons()), "images": count}
