"""
人脸识别模块 - 基于 InsightFace，不可用时回退 OpenCV 简易方案。

流程：
1. 检测人脸位置（InsightFace 或 Haar 级联）
2. 提取特征向量 embedding
3. 与 gallery 里已录入的人脸做余弦相似度比对
"""

from __future__ import annotations

import json     # export_gallery_info 输出 JSON
import pickle   # 序列化 embedding 字典到 .pkl 文件
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import PROJECT_ROOT, load_task_config, resolve_path
from src.core.third_party_paths import INSIGHTFACE_ROOT, bootstrap_env, insightface_model_dir, migrate_insightface_from_user_home


class SimpleFaceBackend:
    """
    InsightFace 不可用时的简易人脸后端。

    检测：OpenCV Haar 级联（传统方法，精度较低但无额外依赖）
    特征：颜色直方图（非深度学习 embedding，仅作演示/回退）
    """

    def __init__(self):
        # cv2.data.haarcascades：OpenCV 自带的级联分类器 XML 目录
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade)

    def get(self, image: np.ndarray) -> list[Any]:
        """
        与 InsightFace FaceAnalysis.get() 接口类似，返回「人脸对象」列表。

        每个对象有 .bbox、.det_score、.embedding 属性。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 级联分类器需要灰度图
        faces = self.detector.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        results = []
        for x, y, w, h in faces:
            crop = image[y : y + h, x : x + w]  # 切片：行 y:y+h，列 x:x+w
            # 8x8x8 的 BGR 颜色直方图作为简易特征
            hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            # 动态创建简单对象，模拟 InsightFace 返回的 face 结构
            obj = type("Face", (), {})()
            obj.bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            obj.det_score = 0.9
            obj.embedding = hist.astype(np.float32)
            results.append(obj)
        return results


class FaceRecognizer:
    """人脸检测 + 识别 + 录入（enroll）。"""

    def __init__(self, model_name: str | None = None):
        cfg = load_task_config("face")
        self.model_name = model_name or cfg.get("model", "buffalo_l")
        self.gallery_dir = resolve_path(cfg.get("gallery", "datasets/face/gallery"))
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.det_threshold = cfg.get("det_threshold", 0.5)
        self._app = None          # InsightFace 或 SimpleFaceBackend 实例
        self._backend = "insightface"
        self._embeddings: dict[str, np.ndarray] = {}  # 姓名 → 特征向量
        self._load_gallery()

    def _get_app(self):
        """懒加载人脸引擎：模型目录 third_party/models/insightface/models/{name}/"""
        if self._app is None:
            bootstrap_env()
            migrate_insightface_from_user_home(self.model_name)
            model_dir = insightface_model_dir(self.model_name)
            if model_dir.exists() and any(model_dir.iterdir()):
                try:
                    from insightface.app import FaceAnalysis

                    self._app = FaceAnalysis(
                        name=self.model_name,
                        root=str(INSIGHTFACE_ROOT),
                        providers=["CPUExecutionProvider"],
                    )
                    self._app.prepare(ctx_id=0, det_size=(640, 640))
                    if hasattr(self._app, "det_model"):
                        self._app.det_model.det_thresh = min(self.det_threshold, 0.3)
                    self._backend = "insightface"
                    print(f"[face] InsightFace 模型: {model_dir}")
                    return self._app
                except Exception as e:
                    print(f"[face] InsightFace 加载失败，回退 OpenCV: {e}")
            else:
                print(
                    f"[face] 未找到模型 {model_dir}，请运行: python scripts/setup_third_party.py --download-face-model"
                )
            self._app = SimpleFaceBackend()
            self._backend = "opencv_simple"
        return self._app

    def _load_gallery(self) -> None:
        """从 embeddings.pkl 加载已录入的人脸特征。"""
        index_path = self.gallery_dir / "embeddings.pkl"
        if index_path.exists():
            with open(index_path, "rb") as f:
                self._embeddings = pickle.load(f)
        else:
            self._embeddings = {}

    def _save_gallery(self) -> None:
        """把内存中的 _embeddings 写回磁盘。"""
        index_path = self.gallery_dir / "embeddings.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(self._embeddings, f)

    def enroll(self, name: str, image_path: str | Path) -> dict[str, Any]:
        """
        录入单张图片为某人的脸。

        成功时更新 _embeddings 并保存参考图到 gallery/{name}/
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return {"success": False, "message": f"无法读取图像: {image_path}"}

        app = self._get_app()
        faces = app.get(img)
        if not faces and self._backend == "insightface":
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

    def enroll_from_dir(self, person_dir: str | Path, name: str | None = None, max_images: int = 20) -> dict[str, Any]:
        """从目录批量录入同一人脸，多张图取特征均值（默认最多 max_images 张）。"""
        person_dir = Path(person_dir)
        person_name = name or person_dir.name
        embs: list[np.ndarray] = []
        count = 0
        img_paths = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )[:max_images]
        for img_path in img_paths:
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

        self._embeddings[person_name] = np.mean(embs, axis=0)  # 多张 embedding 求平均
        self._save_gallery()
        return {"success": True, "name": person_name, "count": count, "backend": self._backend}

    def recognize(self, image: np.ndarray, threshold: float = 0.4) -> list[dict[str, Any]]:
        """
        识别图像中的所有人脸。

        threshold: 相似度高于此才认为匹配到 gallery 中的某人，否则 name='unknown'
        """
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
        """余弦相似度：两向量夹角越小越接近 1。"""
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
