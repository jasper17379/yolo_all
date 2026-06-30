#!/usr/bin/env python3
"""拉取 YOLO 官方源码到 third_party/（便于离线移植与 C++ 对照）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD = PROJECT_ROOT / "third_party"

REPOS = {
    "yolov5": {
        "url": "https://github.com/ultralytics/yolov5.git",
        "branch": "master",
        "desc": "YOLOv5 独立仓库（train.py / export.py / models/）",
    },
    "ultralytics": {
        "url": "https://github.com/ultralytics/ultralytics.git",
        "branch": "main",
        "desc": "Ultralytics 统一仓库（含 YOLOv8 / YOLOv10 训练与导出）",
    },
}


def clone(name: str, force: bool = False) -> bool:
    cfg = REPOS[name]
    dest = THIRD / name
    if dest.exists() and (dest / ".git").exists():
        if not force:
            print(f"[skip] {name} 已存在: {dest}")
            return True
        import shutil

        shutil.rmtree(dest)

    THIRD.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", "--branch", cfg["branch"], cfg["url"], str(dest)]
    print(f"[clone] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="下载 YOLO 源码到 third_party/")
    parser.add_argument("--repo", choices=list(REPOS) + ["all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = list(REPOS) if args.repo == "all" else [args.repo]
    ok = all(clone(n, args.force) for n in names)
    print("\n目录说明:")
    for n, c in REPOS.items():
        print(f"  third_party/{n}/  — {c['desc']}")
    print("\nYOLOv10 模型定义: third_party/ultralytics/ultralytics/cfg/models/v10/")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
