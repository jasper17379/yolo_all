"""人脸识别模块 - 基于 InsightFace，不可用时回退 OpenCV 简易方案"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import PROJECT_ROOT, load_task_config, resolve_path


class SimpleFaceBackend:
    """InsightFace 不可用时的简易人脸后端 (OpenCV Haar + 颜色直方图特征)。"""

    def __init__(self):
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade)

    def get(self, image: np.ndarray) -> list[Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        results = []
        for x, y, w, h in faces:
            crop = image[y : y + h, x : x + w]
            hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            obj = type("Face", (), {})()
            obj.bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            obj.det_score = 0.9
            obj.embedding = hist.astype(np.float32)
            results.append(obj)
        return results


class FaceRecognizer:
    """人脸检测 + 识别 + 录入。"""

    def __init__(self, model_name: str | None = None):
        cfg = load_task_config("face")
        self.model_name = model_name or cfg.get("model", "buffalo_l")
        self.gallery_dir = resolve_path(cfg.get("gallery", "datasets/face/gallery"))
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.det_threshold = cfg.get("det_threshold", 0.5)
        self._app = None
        self._backend = "insightface"
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_gallery()

    def _get_app(self):
        if self._app is None:
            model_dir = Path.home() / ".insightface" / "models" / self.model_name
            if model_dir.exists() and any(model_dir.iterdir()):
                try:
                    from insightface.app import FaceAnalysis

                    self._app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
                    self._app.prepare(ctx_id=0, det_size=(640, 640))
                    if hasattr(self._app, "det_model"):
                        self._app.det_model.det_thresh = min(self.det_threshold, 0.3)
                    self._backend = "insightface"
                    return self._app
                except Exception:
                    pass
            self._app = SimpleFaceBackend()
            self._backend = "opencv_simple"
        return self._app

    def _load_gallery(self) -> None:
        index_path = self.gallery_dir / "embeddings.pkl"
        if index_path.exists():
            with open(index_path, "rb") as f:
                self._embeddings = pickle.load(f)
        else:
            self._embeddings = {}

    def _save_gallery(self) -> None:
        index_path = self.gallery_dir / "embeddings.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(self._embeddings, f)

    def enroll(self, name: str, image_path: str | Path) -> dict[str, Any]:
        """录入新人脸。"""
        img = cv2.imread(str(image_path))
        if img is None:
            return {"success": False, "message": f"无法读取图像: {image_path}"}

        app = self._get_app()
        faces = app.get(img)
        if not faces and self._backend == "insightface":
            # InsightFace 未检出时尝试 OpenCV 回退
            fallback = SimpleFaceBackend()
            faces = fallback.get(img)
            if faces:
                self._backend = "opencv_simple"

        if not faces:
            return {"success": False, "message": "未检测到人脸", "backend": self._backend}

        emb = faces[0].embedding
        self._embeddings[name] = emb
        self._save_gallery()

        person_dir = self.gallery_dir / name
        person_dir.mkdir(exist_ok=True)
        dest = person_dir / Path(image_path).name
        cv2.imwrite(str(dest), img)

        return {"success": True, "name": name, "message": f"已录入人脸: {name}", "backend": self._backend}

    def enroll_from_dir(self, person_dir: str | Path, name: str | None = None) -> dict[str, Any]:
        """从目录批量录入同一人脸，多张图取特征均值。"""
        person_dir = Path(person_dir)
        person_name = name or person_dir.name
        embs: list[np.ndarray] = []
        count = 0
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            app = self._get_app()
            faces = app.get(img)
            if not faces:
                continue
            embs.append(faces[0].embedding)
            count += 1
            person_gallery = self.gallery_dir / person_name
            person_gallery.mkdir(exist_ok=True)
            cv2.imwrite(str(person_gallery / img_path.name), img)

        if not embs:
            return {"success": False, "name": person_name, "count": 0}

        self._embeddings[person_name] = np.mean(embs, axis=0)
        self._save_gallery()
        return {"success": True, "name": person_name, "count": count, "backend": self._backend}

    def recognize(self, image: np.ndarray, threshold: float = 0.4) -> list[dict[str, Any]]:
        """识别图像中的人脸。"""
        app = self._get_app()
        faces = app.get(image)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            item: dict[str, Any] = {
                "bbox": bbox,
                "det_score": float(face.det_score),
                "name": "unknown",
                "similarity": 0.0,
            }
            if self._embeddings:
                best_name, best_sim = "unknown", 0.0
                for name, emb in self._embeddings.items():
                    sim = self._cosine_similarity(face.embedding, emb)
                    if sim > best_sim:
                        best_sim, best_name = sim, name
                if best_sim >= threshold:
                    item["name"] = best_name
                    item["similarity"] = float(best_sim)
            results.append(item)
        return results

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def list_persons(self) -> list[str]:
        return list(self._embeddings.keys())

    def remove_person(self, name: str) -> bool:
        if name in self._embeddings:
            del self._embeddings[name]
            self._save_gallery()
            return True
        return False

    def export_gallery_info(self) -> str:
        info = {"persons": self.list_persons(), "count": len(self._embeddings)}
        return json.dumps(info, ensure_ascii=False, indent=2)
