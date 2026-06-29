#!/usr/bin/env python3
"""端到端验证: 下载数据 -> 训练20轮 -> 推理测试"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd: list[str], desc: str) -> bool:
    print(f"\n{'='*60}\n[{desc}]\n$ {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[FAIL] {desc}")
        return False
    print(f"[OK] {desc}")
    return True


def main():
    report = {"timestamp": datetime.now().isoformat(), "tasks": {}}

    # 1. 下载数据集
    if not run_cmd([sys.executable, "scripts/download_datasets.py", "--all"], "下载/生成数据集"):
        sys.exit(1)

    tasks = ["helmet", "plate", "action", "face"]
    for task in tasks:
        task_report = {"train": None, "infer": None}

        # 2. 训练 20 轮
        train_ok = run_cmd(
            [
                sys.executable, "-m", "src.train.trainer",
                "--task", task,
                "--yolo", "yolov8",
                "--epochs", "20",
                "--batch", "4",
            ],
            f"训练 {task} (20 epochs)",
        )
        task_report["train"] = "ok" if train_ok else "fail"
        if not train_ok:
            report["tasks"][task] = task_report
            continue

        # 3. 推理测试
        if task == "face":
            test_src = PROJECT_ROOT / "datasets" / "face" / "lfw"
            persons = [d for d in test_src.iterdir() if d.is_dir()] if test_src.exists() else []
            if persons:
                imgs = list(persons[0].glob("*.jpg"))
                source = str(imgs[0]) if imgs else str(test_src)
            else:
                source = str(test_src)
        else:
            source = str(PROJECT_ROOT / "datasets" / task / "images" / "val")

        infer_cmd = [
            sys.executable, "-m", "src.infer.inferencer",
            "--task", task,
            "--source", source,
            "--yolo", "yolov8",
            "--output-json", str(PROJECT_ROOT / "outputs" / f"{task}_infer.json"),
        ]
        infer_ok = run_cmd(infer_cmd, f"推理 {task}")
        task_report["infer"] = "ok" if infer_ok else "fail"

        json_path = PROJECT_ROOT / "outputs" / f"{task}_infer.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                task_report["sample_output"] = json.load(f)[:2]

        report["tasks"][task] = task_report

    report_path = PROJECT_ROOT / "outputs" / "verify_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n验证报告: {report_path}")
    all_ok = all(
        t.get("train") == "ok" and t.get("infer") == "ok"
        for t in report["tasks"].values()
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
