"""

统一推理入口。



提供命令行 `python -m src.infer.inferencer` 以及对外的 infer_task 函数，

按任务类型分发到：YOLO 检测 / 人脸识别 / 车牌 OCR。



权重与 --yolo、--model-size 对应，例如:

  --yolo yolov8 --model-size s → weights/{task}/best_yolov8s.pt

"""



from __future__ import annotations



import argparse

import json

from pathlib import Path

from typing import Any



import cv2



from src.core.config import load_task_config, resolve_path
from src.core.third_party_paths import bootstrap_env

from src.core.weights import get_task_weights

from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper





def infer_detection(

    task: str,

    source: str | Path,

    yolo_version: YOLOVersion = "yolov8",

    model_size: str = "n",

    conf: float = 0.25,

    save: bool = True,

) -> list[dict[str, Any]]:

    weights = get_task_weights(task, yolo_version, model_size)

    wrapper = YOLOWrapper(version=yolo_version, weights=weights, model_size=model_size)

    results = wrapper.predict(

        source=source,

        conf=conf,

        project=str(resolve_path("runs/predict")),

        name=f"{task}_{yolo_version}{model_size}",

        save=save,

    )

    outputs = []

    for r in results:

        item: dict[str, Any] = {

            "task": task,

            "weights": str(weights),

            "path": getattr(r, "path", str(source)),

            "detections": [],

        }

        if r.boxes is not None:

            names = r.names

            for box in r.boxes:

                cls_id = int(box.cls[0])

                item["detections"].append(

                    {

                        "class": names[cls_id],

                        "confidence": float(box.conf[0]),

                        "bbox": box.xyxy[0].tolist(),

                    }

                )

        outputs.append(item)

    return outputs





def infer_face(source: str | Path, threshold: float = 0.4) -> list[dict[str, Any]]:

    from src.tasks.face_recognition import FaceRecognizer



    rec = FaceRecognizer()

    img = cv2.imread(str(source))

    if img is None:

        raise FileNotFoundError(f"无法读取图像: {source}")

    faces = rec.recognize(img, threshold=threshold)

    return [{"task": "face", "path": str(source), "faces": faces}]





def infer_plate(

    source: str | Path,

    yolo_version: YOLOVersion = "yolov8",

    model_size: str = "n",

    conf: float = 0.25,

) -> list[dict]:

    from src.tasks.plate_recognition import PlateRecognizer



    rec = PlateRecognizer(yolo_version=yolo_version, model_size=model_size)

    return rec.recognize(source, conf=conf)





def infer_task(

    task: str,

    source: str | Path,

    yolo_version: YOLOVersion = "yolov8",

    model_size: str = "n",

    conf: float = 0.25,

    save: bool = True,

) -> list[dict[str, Any]]:

    cfg = load_task_config(task)

    if cfg.get("type") == "recognition":

        return infer_face(source, threshold=cfg.get("rec_threshold", 0.4))

    if task == "plate":

        return infer_plate(source, yolo_version=yolo_version, model_size=model_size, conf=conf)

    return infer_detection(task, source, yolo_version, model_size, conf, save)





def main():

    bootstrap_env()

    parser = argparse.ArgumentParser(description="Vision AI 统一推理")

    parser.add_argument("--task", required=True, choices=["helmet", "plate", "action", "face"])

    parser.add_argument("--source", required=True, help="图像/目录/视频路径")

    parser.add_argument("--yolo", default="yolov8", choices=["yolov5", "yolov8", "yolov10"])

    parser.add_argument(

        "--model-size",

        default="n",

        choices=["n", "s", "m", "l", "x", "b"],

        help="须与训练时一致，如 best_yolov8s.pt 对应 --model-size s",

    )

    parser.add_argument("--conf", type=float, default=0.25)

    parser.add_argument("--no-save", action="store_true")

    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()



    results = infer_task(

        task=args.task,

        source=resolve_path(args.source),

        yolo_version=args.yolo,

        model_size=args.model_size,

        conf=args.conf,

        save=not args.no_save,

    )

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output_json:

        out = Path(args.output_json)

        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:

            json.dump(results, f, ensure_ascii=False, indent=2)





if __name__ == "__main__":

    main()


