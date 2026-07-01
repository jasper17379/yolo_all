"""USB 摄像头动作识别快照（非交互，保存标注帧供查看效果）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_task_config
from src.core.device import add_device_arg, resolve_yolo_device
from src.core.third_party_paths import bootstrap_env
from src.core.weights import get_task_weights
from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper
from src.core.cv_draw import draw_label_bgr

bootstrap_env()
BOX_COLOR = (0, 0, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description="USB 摄像头动作识别快照")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30, help="预热+采样帧数")
    parser.add_argument("--out", default="runs/demo/action_camera_snapshot.jpg")
    parser.add_argument("--yolo", default="yolov8", choices=["yolov5", "yolov8", "yolov10"])
    parser.add_argument("--model-size", default="n")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=416)
    add_device_arg(parser, default="auto")
    args = parser.parse_args()

    task_cfg = load_task_config("action")
    classes = task_cfg.get("classes", {})
    if isinstance(classes, list):
        classes = {i: n for i, n in enumerate(classes)}

    weights = get_task_weights("action", args.yolo, args.model_size)
    wrapper = YOLOWrapper(version=args.yolo, weights=str(weights), model_size=args.model_size)
    device = resolve_yolo_device(args.device)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")

    best_frame = None
    best_dets: list[dict] = []
    print(f"摄像头 {args.camera} 已打开，采样 {args.frames} 帧...")

    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        if i < 5:
            continue  # 预热
        results = wrapper.predict(source=frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)
        dets: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dets.append(
                    {
                        "label": classes.get(cls_id, str(cls_id)),
                        "conf": float(box.conf[0]),
                        "bbox": (x1, y1, x2, y2),
                    }
                )
        if len(dets) >= len(best_dets):
            best_frame = frame.copy()
            best_dets = dets

    cap.release()

    if best_frame is None:
        raise RuntimeError("未从摄像头读取到有效帧")

    out = best_frame.copy()
    for d in best_dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        text = f"action:{d['label']} {d['conf']:.2f}"
        out = draw_label_bgr(out, text, (x1, y1), box_color_bgr=BOX_COLOR)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"已保存: {out_path}")
    if best_dets:
        for d in best_dets:
            print(f"  - {d['label']} {d['conf']:.2f}")
    else:
        print("  (未检测到动作，请对着摄像头做手势后重试)")


if __name__ == "__main__":
    main()
