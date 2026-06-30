"""
REST API - 训练数据添加、人脸录入、推理。

启动方式:
  python -m src.api.server
  或 uvicorn src.api.server:app --host 0.0.0.0 --port 8000

依赖:
  fastapi  - Web 框架，定义路由和请求体
  pydantic - 数据校验（TrainRequest / InferRequest）
  uvicorn  - ASGI 服务器，运行 FastAPI 应用
"""

from __future__ import annotations

import shutil  # 保存上传文件
import uuid    # 生成唯一文件名，避免覆盖
from pathlib import Path
from typing import Optional  # Optional[str] 等价于 str | None（旧写法）

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.config import PROJECT_ROOT, ensure_dirs
from src.infer.inferencer import infer_task
from src.tasks.face_recognition import FaceRecognizer
from src.train.trainer import train_task

# 创建 FastAPI 应用实例，title/version 会出现在自动文档 /docs 里
app = FastAPI(title="Vision AI Platform API", version="1.0.0")


class TrainRequest(BaseModel):
    """
    Pydantic 模型：定义 POST /api/v1/train 的 JSON  body 字段和类型。

    缺少必填字段或类型不对时，FastAPI 自动返回 422 错误。
    """

    task: str
    yolo_version: str = "yolov8"
    model_size: str = "n"
    epochs: int = 20
    batch: int = 8
    resume_from_best: bool = False
    weights: Optional[str] = None


class InferRequest(BaseModel):
    """推理 API 的请求体。"""

    task: str
    source: str
    yolo_version: str = "yolov8"
    model_size: str = "n"
    conf: float = 0.25


@app.on_event("startup")
def startup():
    """服务启动时执行一次，确保目录存在。"""
    ensure_dirs()


def create_app() -> FastAPI:
    """工厂函数，便于测试或其他模块挂载 app。"""
    return app


@app.get("/health")
def health():
    """健康检查，负载均衡或监控常用。"""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/v1/train")
def api_train(req: TrainRequest):
    """触发模型训练（同步执行，耗时会较长）。"""
    try:
        result = train_task(
            task=req.task,
            yolo_version=req.yolo_version,
            model_size=req.model_size,
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
    """对指定路径的图片/目录做推理。"""
    try:
        results = infer_task(req.task, req.source, req.yolo_version, req.model_size, req.conf)
        return {"success": True, "results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/v1/face/enroll")
async def face_enroll(name: str = Form(...), image: UploadFile = File(...)):
    """
    录入新人脸（multipart/form-data 上传）。

    Form(...): 必填表单字段
    UploadFile: 上传的文件流
    async: 异步路由，适合 I/O 密集的上传处理
    """
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
    """列出 gallery 中已录入的姓名。"""
    rec = FaceRecognizer()
    return {"persons": rec.list_persons(), "count": len(rec.list_persons())}


@app.delete("/api/v1/face/{name}")
def face_delete(name: str):
    """从 gallery 删除某人（路径参数 name）。"""
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
    """
    添加自定义训练数据到 datasets/custom/。

    detection 任务 label 为 YOLO 格式一行: class_id x_center y_center width height（归一化 0~1）
    若只传 class_id，则写入默认居中框 0.5 0.5 0.9 0.9
    """
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
    """
    启动 uvicorn 服务器。

    0.0.0.0 表示监听所有网卡，局域网其他机器可访问。
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
