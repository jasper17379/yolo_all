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

# ---------------------------------------------------------------------------
# 标准库导入（Python 自带，无需 pip install）
# ---------------------------------------------------------------------------

# from __future__ import annotations
# 启用「延迟类型注解」：允许在类定义完成前就引用类名（如 RealtimeDemo | None）。
# Python 3.7+ 可选特性，3.10+ 默认行为类似，保留此导入是为了兼容旧版本。
from __future__ import annotations

# argparse：解析命令行参数，例如 --tasks helmet --camera 0
import argparse

# atexit：注册「程序正常退出时」要执行的清理函数（类似 C 的 atexit）
import atexit

# gc：垃圾回收模块，cleanup 时手动触发，帮助释放大模型占用的内存
import gc

# signal：捕获 Ctrl+C (SIGINT)、kill (SIGTERM) 等信号，优雅退出
import signal

# sys：解释器相关，如 sys.path（模块搜索路径）、sys.exit（退出进程）
import sys

# time：时间相关，这里用来计算 FPS（每秒帧数）
import time

# pathlib.Path：面向对象的文件/目录路径，比字符串拼接路径更安全、跨平台
from pathlib import Path

# ---------------------------------------------------------------------------
# 第三方库导入（需要先 pip install）
# ---------------------------------------------------------------------------

# cv2：OpenCV 的 Python 接口，用于读摄像头、显示窗口、画框、写文字
import cv2

# numpy（常写作 np）：数值计算库，图像在内存里是 ndarray 数组
import numpy as np

# 禁止 Python 在本目录生成 .pyc 字节码缓存文件（开发时目录更干净）
sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# 项目路径设置
# ---------------------------------------------------------------------------

# Path(__file__)：当前脚本文件的路径
# .resolve()：转成绝对路径
# .parent：上一级目录，即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 把项目根目录插入 sys.path 最前面，这样 import src.xxx 才能找到本项目的包
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 本项目内部模块
# ---------------------------------------------------------------------------

# load_task_config：读取 configs/tasks/{task}.yaml 里的任务配置（类别名等）
from src.core.config import load_global_config, load_task_config
from src.core.device import add_device_arg, print_device_info, resolve_yolo_device

# YOLOWrapper：对 Ultralytics YOLO 的统一封装，负责加载权重和 predict
from src.core.yolo_wrapper import YOLOWrapper

# get_task_weights：按任务名查找 best.pt 权重文件路径
from src.infer.inferencer import get_task_weights

# FaceRecognizer：人脸检测 + 与 gallery 比对识别
from src.tasks.face_recognition import FaceRecognizer

# ---------------------------------------------------------------------------
# 常量：窗口名、画框颜色、字体
# ---------------------------------------------------------------------------

WINDOW_NAME = "Vision AI Demo [Q=quit]"

# OpenCV 颜色是 BGR 顺序：(蓝, 绿, 红)，(0,0,255) = 红色
BOX_COLOR = (0, 0, 255)

# Hershey 简单字体，用于在画面上叠加文字
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 全局变量：保存当前运行的 Demo 实例，供信号处理函数强制释放摄像头
_active_demo: "RealtimeDemo | None" = None


class RealtimeDemo:
    """
    实时摄像头推理主类。

    流程概览：
    1. __init__：加载 YOLO 模型、可选人脸模型、打开摄像头
    2. run()：循环读帧 → 推理 → 画框 → imshow
    3. cleanup()：释放摄像头和窗口
    """

    def __init__(
        self,
        tasks: list[str],           # 要启用的任务列表，如 ["helmet", "face"]
        camera_id: int = 0,           # 摄像头设备号，0 通常是默认 USB 摄像头
        yolo_version: str = "yolov8", # YOLO 版本：v5 / v8 / v10
        model_size: str = "n",        # 模型规模 n/s/m/l/x，对应 best_yolov8n.pt 等
        device: str = "auto",         # auto | cpu | cuda:0 | 0 | 0,1
        conf: float = 0.35,           # 检测置信度阈值，低于此值的框会被丢弃
        use_face: bool = True,        # 是否启用人脸识别
        imgsz: int = 416,             # 推理时图像缩放到该尺寸（越小越快）
        face_interval: int = 5,       # 人脸每 N 帧才推理一次，降低 CPU 占用
    ):
        self.tasks = tasks
        self.conf = conf
        self.imgsz = imgsz
        self.yolo_version = yolo_version
        self.model_size = model_size
        self.device = device
        self.yolo_device = resolve_yolo_device(device)
        self.face_interval = max(1, face_interval)  # 至少为 1，避免除零
        self._running = False       # 主循环是否继续
        self._frame_idx = 0         # 当前帧序号，用于 face_interval
        self._last_face_dets: list[dict] = []  # 上一帧人脸结果，跳帧时复用

        # dict[str, YOLOWrapper]：任务名 → 对应 YOLO 模型实例
        self.models: dict[str, YOLOWrapper] = {}
        # 每个任务的类别 id → 类别名称（用于显示标签）
        self.class_names: dict[str, dict] = {}

        # 为每个非 face 任务加载 YOLO 权重和配置
        for task in tasks:
            if task == "face":
                continue  # 人脸走 FaceRecognizer，不用 YOLO
            weights = get_task_weights(task, yolo_version, model_size)
            self.models[task] = YOLOWrapper(version=yolo_version, weights=weights, model_size=model_size)
            cfg = load_task_config(task)
            names = cfg.get("classes") or {}
            # YAML 里 classes 可能是 dict 或 list，统一转成 {id: name}
            self.class_names[task] = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}

        self.face_rec: FaceRecognizer | None = None
        if use_face and "face" in tasks:
            self.face_rec = FaceRecognizer(device=device)

        self.plate_rec = None
        if "plate" in tasks:
            from src.tasks.plate_recognition import PlateRecognizer

            self.plate_rec = PlateRecognizer(
                yolo_version=yolo_version,
                model_size=model_size,
                device=device,
            )
            # plate 走 PlateRecognizer（YOLO+OCR），不再单独加载到 models
            self.models.pop("plate", None)

        # 打开摄像头：Windows 优先用 DirectShow (CAP_DSHOW)，兼容性更好
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)  # 回退默认后端
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        # 设置分辨率与缓冲区（缓冲区=1 可减少延迟，总是取最新帧）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def stop(self) -> None:
        """请求停止主循环（把 _running 设为 False）。"""
        self._running = False

    def cleanup(self) -> None:
        """释放摄像头、窗口、模型，避免进程/终端挂死。"""
        self._running = False

        # 释放 VideoCapture
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.cap = None

        # 多次 destroy 确保 OpenCV 窗口在 Windows 上真正关闭
        for _ in range(5):
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # 给 GUI 线程处理关闭事件的机会

        self.models.clear()
        self.face_rec = None
        self.plate_rec = None
        gc.collect()

        # 若安装了 PyTorch 且用 GPU，清空 CUDA 缓存
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _predict_yolo(self, task: str, frame: np.ndarray) -> list[dict]:
        """
        对单帧做 YOLO 检测。

        参数:
            task: 任务名（helmet / plate / action）
            frame: BGR 图像数组，形状一般为 (高, 宽, 3)

        返回:
            检测结果列表，每项含 task、label、conf、bbox
        """
        if not self._running:
            return []
        wrapper = self.models.get(task)
        if wrapper is None:
            return []

        results = wrapper.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.yolo_device,
            save=False,
            verbose=False,
        )

        dets = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names  # 模型内置类别名
            for box in r.boxes:
                cls_id = int(box.cls[0])   # 类别 id
                # xyxy：左上角 (x1,y1)、右下角 (x2,y2)
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

    def _predict_plate(self, frame: np.ndarray) -> list[dict]:
        """车牌：YOLO 定位 + HyperLPR3/PaddleOCR 读字。"""
        if not self._running or not self.plate_rec:
            return []
        try:
            plates = self.plate_rec.recognize_frame(frame, conf=self.conf, imgsz=self.imgsz)
        except Exception as e:
            print(f"[plate] 推理异常(已跳过): {e}")
            return []
        dets = []
        for p in plates:
            bbox = p.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            text = p.get("plate_text") or "plate"
            ocr_conf = p.get("ocr_confidence", 0)
            label = f"{text}({ocr_conf:.2f})" if text != "plate" else "plate"
            dets.append(
                {
                    "task": "plate",
                    "label": label,
                    "conf": float(p.get("confidence", 0)),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )
        return dets

    def _predict_face(self, frame: np.ndarray) -> list[dict]:
        """人脸识别；失败时返回上一帧结果，避免画面闪烁。"""
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
        """
        在图像上画矩形框和标签。

        @staticmethod：不依赖 self，可直接 RealtimeDemo._draw_detections(...) 调用
        frame.copy()：复制一份再画，避免修改原始帧
        """
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
            text = f"{d['task']}:{d['label']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
            # 文字背景条（实心矩形 -1 表示填充）
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), BOX_COLOR, -1)
            cv2.putText(out, text, (x1, y1 - 4), FONT, 0.5, (255, 255, 255), 1)
        return out

    @staticmethod
    def _window_closed() -> bool:
        """检查 OpenCV 窗口是否已被用户点击关闭。"""
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            return True

    def run(self) -> None:
        """主循环：读摄像头 → 多任务推理 → 显示。"""
        global _active_demo
        _active_demo = self
        self._running = True

        print("按 Q / ESC / 关窗口 / Ctrl+C 退出")
        print(f"启用: {', '.join(self.tasks)} | 人脸间隔: 每 {self.face_interval} 帧")
        fps_t = time.time()
        frame_count = 0

        try:
            while self._running:
                # ret：是否成功；frame：numpy 图像
                ret, frame = self.cap.read()
                if not ret:
                    print("[warn] 摄像头读取失败，退出")
                    break

                detections: list[dict] = []

                # YOLO 任务逐个推理
                for task in self.tasks:
                    if task == "face" or not self._running:
                        continue
                    if task == "plate":
                        detections.extend(self._predict_plate(frame))
                    else:
                        detections.extend(self._predict_yolo(task, frame))

                # 人脸：按间隔推理，中间帧复用 _last_face_dets
                if self.face_rec and "face" in self.tasks:
                    if self._frame_idx % self.face_interval == 0:
                        detections.extend(self._predict_face(frame))
                    else:
                        detections.extend(self._last_face_dets)

                self._frame_idx += 1
                display = self._draw_detections(frame, detections)

                # 每 10 帧更新一次 FPS 显示
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
                # waitKey(1)：等待 1ms 并读取按键；& 0xFF 取低 8 位（兼容某些平台）
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                    print("用户退出")
                    break
                if self._window_closed():
                    print("窗口已关闭")
                    break
        except KeyboardInterrupt:
            print("\nCtrl+C 已中断")
        finally:
            # finally：无论正常退出还是异常，都会执行 cleanup
            self.cleanup()
            _active_demo = None
            print("资源已释放，进程可正常退出")


def _cleanup_only() -> None:
    """仅做资源清理，供 atexit / 信号处理调用。"""
    global _active_demo
    if _active_demo is not None:
        _active_demo.cleanup()
        _active_demo = None


def _signal_handler(signum, frame) -> None:
    """
    信号处理函数。signum 为信号编号，frame 为当前栈帧（通常不用）。

    128 + signum 是 Unix 惯例：脚本因信号退出时的返回码。
    """
    print("\n收到退出信号，正在释放摄像头和模型...")
    _cleanup_only()
    sys.exit(128 + signum)


def main():
    """程序入口：注册退出钩子 → 解析命令行 → 创建 Demo 并 run。"""
    from src.core.third_party_paths import bootstrap_env

    bootstrap_env()

    # 注册 Ctrl+C 和 kill 的处理
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    # 进程正常结束时也尝试 cleanup（双保险）
    atexit.register(_cleanup_only)

    parser = argparse.ArgumentParser(description="USB摄像头实时多模型推理")
    parser.add_argument(
        "--tasks",
        nargs="+",  # 至少一个，可多个：--tasks helmet plate
        default=["helmet", "plate", "action", "face"],
        choices=["helmet", "plate", "action", "face"],
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--yolo", default="yolov8", choices=["yolov5", "yolov8", "yolov10"])
    add_device_arg(parser, default=load_global_config().get("device", "auto"))
    parser.add_argument(
        "--model-size",
        default="n",
        choices=["n", "s", "m", "l", "x", "b"],
        help="YOLO 模型规模，须与 weights/{task}/best_{yolo}{size}.pt 一致",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--no-face", action="store_true")  # 出现即 True，无需传值
    parser.add_argument("--face-interval", type=int, default=5, help="人脸每N帧推理一次(降CPU)")
    parser.add_argument(
        "--lite",
        action="store_true",
        help="轻量模式: 仅 helmet + 关闭 face",
    )
    args = parser.parse_args()
    print_device_info(args.device)

    tasks = args.tasks
    if args.lite:
        tasks = ["helmet"]
        args.no_face = True
    if args.no_face:
        tasks = [t for t in tasks if t != "face"]  # 列表推导式：过滤掉 face

    demo = None
    try:
        demo = RealtimeDemo(
            tasks=tasks,
            camera_id=args.camera,
            yolo_version=args.yolo,
            model_size=args.model_size,
            device=args.device,
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


# Python 约定：直接运行本文件时 __name__ == "__main__"，被 import 时为模块名
if __name__ == "__main__":
    main()
