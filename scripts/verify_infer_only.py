#!/usr/bin/env python3
"""仅运行推理验证 (训练已完成时使用)"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_infer(task: str, source: str) -> dict:
    out_json = PROJECT_ROOT / "outputs" / f"{task}_infer.json"
    cmd = [
        sys.executable, "-m", "src.infer.inferencer",
        "--task", task, "--source", source, "--yolo", "yolov8",
        "--output-json", str(out_json),
    ]
    print(f"\n>>> 推理 {task}: {source}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    result = {"task": task, "status": "ok" if r.returncode == 0 else "fail"}
    if out_json.exists():
        with open(out_json, encoding="utf-8") as f:
            data = json.load(f)
            result["sample"] = data[:1] if isinstance(data, list) else data
    return result


def main():
    report = {"timestamp": datetime.now().isoformat(), "infer": {}}

    # 人脸训练 (gallery 构建)
    print("\n>>> 人脸库构建")
    r = subprocess.run(
        [sys.executable, "-m", "src.train.trainer", "--task", "face"],
        cwd=str(PROJECT_ROOT),
    )
    report["face_train"] = "ok" if r.returncode == 0 else "fail"

    tasks = {
        "helmet": str(PROJECT_ROOT / "datasets" / "helmet" / "images" / "val"),
        "plate": str(PROJECT_ROOT / "datasets" / "plate" / "images" / "val"),
        "action": str(PROJECT_ROOT / "datasets" / "action" / "images" / "val"),
    }
    lfw = PROJECT_ROOT / "datasets" / "face" / "lfw"
    persons = [d for d in lfw.iterdir() if d.is_dir()] if lfw.exists() else []
    face_src = str(list(persons[0].glob("*.jpg"))[0]) if persons else str(lfw)
    tasks["face"] = face_src

    for task, source in tasks.items():
        report["infer"][task] = run_infer(task, source)

    report_path = PROJECT_ROOT / "outputs" / "verify_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 合并已有训练结果
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            old = json.load(f)
        old["infer_rerun"] = report
        report = old
        report["infer"] = report.get("infer_rerun", {}).get("infer", {})
        for t in ["helmet", "plate", "action", "face"]:
            if t in report.get("infer_rerun", {}).get("infer", {}):
                report.setdefault("tasks", {})[t] = report["tasks"].get(t, {})
                report["tasks"][t]["infer"] = report["infer_rerun"]["infer"][t]["status"]
                report["tasks"][t]["train"] = report["tasks"][t].get("train", "ok")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    all_ok = all(v.get("status") == "ok" for v in report.get("infer_rerun", report).get("infer", report.get("infer", {})).values())
    print(f"\n验证报告: {report_path}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
