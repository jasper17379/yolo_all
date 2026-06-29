#!/usr/bin/env python3
"""
USB 摄像头实时多模型推理 Demo。
支持: 安全帽 / 车牌 / 动作 / 人脸识别，检测目标画红框。

用法:
  python demo.py                    # 默认 helmet+plate+action+face
  python demo.py --tasks helmet --no-face   # 轻量，仅安全帽
  python demo.py --lite             # 推荐：仅 helmet，人脸每5帧一次

按 Q / ESC / 关闭窗口 / Ctrl+C 均可退出，资源会自动释放。
"""

from __future__ import annotations

import argparse
import atexit
import gc
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_task_config
from src.core.yolo_wrapper import YOLOWrapper
from src.infer.inferencer import get_task_weights
from src.tasks.face_recognition import FaceRecognizer

WINDOW_NAME = "Vision AI Demo [Q=quit]"
BOX_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 全局实例，供信号处理强制清理
_active_demo: "RealtimeDemo | None" = None


class RealtimeDemo:
    def __init__(
        self,
        tasks: list[str],
        camera_id: int = 0,
        yolo_version: str = "yolov8",
        conf: float = 0.35,
        use_face: bool = True,
        imgsz: int = 416,
        face_interval: int = 5,
    ):
        self.tasks = tasks
        self.conf = conf
        self.imgsz = imgsz
        self.face_interval = max(1, face_interval)
        self._running = False
        self._frame_idx = 0
        self._last_face_dets: list[dict] = []

        self.models: dict[str, YOLOWrapper] = {}
        self.class_names: dict[str, dict] = {}

        for task in tasks:
            if task == "face":
                continue
            weights = get_task_weights(task)
            self.models[task] = YOLOWrapper(version=yolo_version, weights=weights)
            cfg = load_task_config(task)
            names = cfg.get("classes") or {}
            self.class_names[task] = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}

        self.face_rec: FaceRecognizer | None = None
        if use_face and "face" in tasks:
            self.face_rec = FaceRecognizer()

        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def stop(self) -> None:
        """请求停止主循环。"""
        self._running = False

    def cleanup(self) -> None:
        """释放摄像头、窗口、模型，避免进程/终端挂死。"""
        self._running = False
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.cap = None

        for _ in range(5):
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        self.models.clear()
        self.face_rec = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _predict_yolo(self, task: str, frame: np.ndarray) -> list[dict]:
        if not self._running:
            return []
        wrapper = self.models.get(task)
        if wrapper is None:
            return []
        results = wrapper.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            save=False,
            verbose=False,
        )
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = self.class_names[task].get(cls_id, names.get(cls_id, str(cls_id)))
                dets.append(
                    {
                        "task": task,
                        "label": label,
                        "conf": float(box.conf[0]),
                        "bbox": (x1, y1, x2, y2),
                    }
                )
        return dets

    def _predict_face(self, frame: np.ndarray) -> list[dict]:
        if not self._running or not self.face_rec:
            return self._last_face_dets
        try:
            faces = self.face_rec.recognize(frame, threshold=0.4)
        except Exception as e:
            print(f"[face] 推理异常(已跳过): {e}")
            return self._last_face_dets
        dets = []
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            name = f.get("name", "unknown")
            sim = f.get("similarity", 0)
            label = f"{name}({sim:.2f})" if name != "unknown" else "face"
            dets.append(
                {
                    "task": "face",
                    "label": label,
                    "conf": f.get("det_score", 0),
                    "bbox": (x1, y1, x2, y2),
                }
            )
        self._last_face_dets = dets
        return dets

    @staticmethod
    def _draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
            text = f"{d['task']}:{d['label']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), BOX_COLOR, -1)
            cv2.putText(out, text, (x1, y1 - 4), FONT, 0.5, (255, 255, 255), 1)
        return out

    @staticmethod
    def _window_closed() -> bool:
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            return True

    def run(self) -> None:
        global _active_demo
        _active_demo = self
        self._running = True

        print("按 Q / ESC / 关窗口 / Ctrl+C 退出")
        print(f"启用: {', '.join(self.tasks)} | 人脸间隔: 每 {self.face_interval} 帧")
        fps_t = time.time()
        frame_count = 0

        try:
            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[warn] 摄像头读取失败，退出")
                    break

                detections: list[dict] = []
                for task in self.tasks:
                    if task == "face" or not self._running:
                        continue
                    detections.extend(self._predict_yolo(task, frame))

                if self.face_rec and "face" in self.tasks:
                    if self._frame_idx % self.face_interval == 0:
                        detections.extend(self._predict_face(frame))
                    else:
                        detections.extend(self._last_face_dets)

                self._frame_idx += 1
                display = self._draw_detections(frame, detections)

                frame_count += 1
                if frame_count % 10 == 0:
                    now = time.time()
                    fps = 10 / max(now - fps_t, 1e-6)
                    fps_t = now
                    cv2.putText(
                        display,
                        f"FPS:{fps:.1f} det:{len(detections)}",
                        (10, 25),
                        FONT,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("用户退出")
                    break
                if self._window_closed():
                    print("窗口已关闭")
                    break
        except KeyboardInterrupt:
            print("\nCtrl+C 已中断")
        finally:
            self.cleanup()
            _active_demo = None
            print("资源已释放，进程可正常退出")


def _cleanup_only() -> None:
    global _active_demo
    if _active_demo is not None:
        _active_demo.cleanup()
        _active_demo = None


def _signal_handler(signum, frame) -> None:
    print("\n收到退出信号，正在释放摄像头和模型...")
    _cleanup_only()
    sys.exit(128 + signum)


def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup_only)

    parser = argparse.ArgumentParser(description="USB摄像头实时多模型推理")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["helmet", "plate", "action", "face"],
        choices=["helmet", "plate", "action", "face"],
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--yolo", default="yolov8", choices=["yolov5", "yolov8", "yolov10"])
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--face-interval", type=int, default=5, help="人脸每N帧推理一次(降CPU)")
    parser.add_argument(
        "--lite",
        action="store_true",
        help="轻量模式: 仅 helmet + 关闭 face",
    )
    args = parser.parse_args()

    tasks = args.tasks
    if args.lite:
        tasks = ["helmet"]
        args.no_face = True
    if args.no_face:
        tasks = [t for t in tasks if t != "face"]

    demo = None
    try:
        demo = RealtimeDemo(
            tasks=tasks,
            camera_id=args.camera,
            yolo_version=args.yolo,
            conf=args.conf,
            use_face="face" in tasks,
            imgsz=args.imgsz,
            face_interval=args.face_interval,
        )
        demo.run()
    except KeyboardInterrupt:
        _cleanup_only()
    except Exception as e:
        print(f"错误: {e}")
        if demo is not None:
            demo.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
