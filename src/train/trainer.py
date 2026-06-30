"""

统一训练入口。



命令行示例:

  python -m src.train.trainer --task helmet --epochs 20

  python -m src.train.trainer --task plate --yolo yolov8 --model-size s --epochs 20

  python -m src.train.trainer --task plate --resume-from-best --yolo yolov8 --model-size s

"""



from __future__ import annotations



import argparse

from pathlib import Path



from src.core.config import PROJECT_ROOT, ensure_dirs, load_global_config, load_task_config, resolve_path
from src.core.dataset_yaml import prepare_data_yaml
from src.core.third_party_paths import bootstrap_env

from src.core.weights import find_trained_best, get_task_weights, resolve_pretrained, run_name, sync_trained_best

from src.core.yolo_wrapper import YOLOVersion, YOLOWrapper





def train_task(

    task: str,

    yolo_version: YOLOVersion = "yolov8",

    model_size: str = "n",

    epochs: int = 20,

    batch: int = 8,

    imgsz: int = 640,

    resume_from_best: bool = False,

    weights: str | None = None,

    resume: bool = False,

) -> dict:

    """

    训练指定任务。



    resume_from_best: 在 weights/{task}/best_{yolov8n}.pt 基础上继续训练

    resume: Ultralytics 断点续训（从 last.pt 同一 run 继续）

    """

    ensure_dirs()
    bootstrap_env()
    task_cfg = load_task_config(task)

    global_cfg = load_global_config()



    if task_cfg.get("type") == "recognition":
        from src.tasks.face_trainer import train_face_gallery

        return train_face_gallery(max_images_per_person=task_cfg.get("max_images_per_person", 20))



    data_yaml = prepare_data_yaml(task_cfg["dataset"])



    if weights:

        init_weights: str | Path = Path(weights) if Path(weights).exists() else weights

    elif resume_from_best:

        best_path = find_trained_best(task, yolo_version, model_size)

        if best_path:

            init_weights = best_path

            print(f"[{task}] 基于已训练 best 继续: {best_path}")

        else:

            init_weights = resolve_pretrained(yolo_version, model_size)

            print(f"[{task}] 未找到 best_{yolo_version}{model_size}，改用预训练: {init_weights}")

    else:

        init_weights = resolve_pretrained(yolo_version, model_size)

        print(f"[{task}] 从预训练权重开始: {init_weights}")



    wrapper = YOLOWrapper(version=yolo_version, weights=str(init_weights), model_size=model_size)



    exp_name = run_name(task, yolo_version, model_size)

    result = wrapper.train(

        data_yaml=data_yaml,

        epochs=epochs,

        batch=batch,

        imgsz=imgsz,

        project=str(PROJECT_ROOT / "runs" / "train"),

        name=exp_name,

        resume=resume,

    )



    dest = sync_trained_best(task, yolo_version, model_size, result["best"])

    print(f"[{task}] 训练完成, best 权重: {dest}")

    print(f"[{task}] 原始 run 目录: {result['save_dir']}")

    return result





def main():

    parser = argparse.ArgumentParser(description="Vision AI 统一训练")

    parser.add_argument("--task", required=True, choices=["helmet", "plate", "action", "face", "all"])

    parser.add_argument("--yolo", default="yolov8", choices=["yolov5", "yolov8", "yolov10"])

    parser.add_argument(

        "--model-size",

        default="n",

        choices=["n", "s", "m", "l", "x", "b"],

        help="模型规模: n/s/m/l/x (yolov10 另有 b)",

    )

    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--batch", type=int, default=8)

    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument(

        "--resume-from-best",

        action="store_true",

        help="在 weights/{task}/best_{yolo}{size}.pt 基础上继续训练",

    )

    parser.add_argument(

        "--weights",

        type=str,

        default=None,

        help="指定预训练权重路径（优先于 --model-size）",

    )

    parser.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")

    args = parser.parse_args()



    tasks = ["helmet", "plate", "action", "face"] if args.task == "all" else [args.task]

    for t in tasks:

        train_task(

            task=t,

            yolo_version=args.yolo,

            model_size=args.model_size,

            epochs=args.epochs,

            batch=args.batch,

            imgsz=args.imgsz,

            resume_from_best=args.resume_from_best,

            weights=args.weights,

            resume=args.resume,

        )





if __name__ == "__main__":

    main()


