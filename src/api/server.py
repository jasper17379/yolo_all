"""REST API - 训练数据添加、人脸录入、推理"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.config import PROJECT_ROOT, ensure_dirs
from src.infer.inferencer import infer_task
from src.tasks.face_recognition import FaceRecognizer
from src.train.trainer import train_task

app = FastAPI(title="Vision AI Platform API", version="1.0.0")


class TrainRequest(BaseModel):
    task: str
    yolo_version: str = "yolov8"
    epochs: int = 20
    batch: int = 8
    resume_from_best: bool = False
    weights: Optional[str] = None


class InferRequest(BaseModel):
    task: str
    source: str
    yolo_version: str = "yolov8"
    conf: float = 0.25


@app.on_event("startup")
def startup():
    ensure_dirs()


def create_app() -> FastAPI:
    return app


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/v1/train")
def api_train(req: TrainRequest):
    try:
        result = train_task(
            task=req.task,
            yolo_version=req.yolo_version,
            epochs=req.epochs,
            batch=req.batch,
            resume_from_best=req.resume_from_best,
            weights=req.weights,
        )
        return {"success": True, "result": {"best": result.get("best"), "save_dir": result.get("save_dir")}}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/v1/infer")
def api_infer(req: InferRequest):
    try:
        results = infer_task(req.task, req.source, req.yolo_version, req.conf)
        return {"success": True, "results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/v1/face/enroll")
async def face_enroll(name: str = Form(...), image: UploadFile = File(...)):
    """录入新人脸。"""
    try:
        save_dir = PROJECT_ROOT / "datasets" / "face" / "gallery" / "uploads"
        save_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(image.filename or "img.jpg").suffix or ".jpg"
        save_path = save_dir / f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        rec = FaceRecognizer()
        result = rec.enroll(name, save_path)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/v1/face/list")
def face_list():
    rec = FaceRecognizer()
    return {"persons": rec.list_persons(), "count": len(rec.list_persons())}


@app.delete("/api/v1/face/{name}")
def face_delete(name: str):
    rec = FaceRecognizer()
    ok = rec.remove_person(name)
    return {"success": ok}


@app.post("/api/v1/data/add")
async def add_training_data(
    task: str = Form(...),
    image: UploadFile = File(...),
    label: Optional[str] = Form(None),
    class_id: Optional[int] = Form(None),
):
    """添加自定义训练数据。detection 任务需提供 class_id 和 YOLO 格式 label。"""
    try:
        custom_img = PROJECT_ROOT / "datasets" / "custom" / "images" / task
        custom_lbl = PROJECT_ROOT / "datasets" / "custom" / "labels" / task
        custom_img.mkdir(parents=True, exist_ok=True)
        custom_lbl.mkdir(parents=True, exist_ok=True)

        stem = uuid.uuid4().hex[:12]
        ext = Path(image.filename or "img.jpg").suffix or ".jpg"
        img_path = custom_img / f"{stem}{ext}"
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        if label:
            lbl_path = custom_lbl / f"{stem}.txt"
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(label)
        elif class_id is not None:
            lbl_path = custom_lbl / f"{stem}.txt"
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(f"{class_id} 0.5 0.5 0.9 0.9")

        return {"success": True, "image": str(img_path), "task": task}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
