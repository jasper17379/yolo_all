#!/usr/bin/env python3
"""下载 YOLOv5 / v8 / v10 常用预训练权重到 weights/pretrained/。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.third_party_paths import bootstrap_env
from src.core.weights import PRETRAINED_DIR, list_pretrained_names


def download_one(name: str, dest_dir: Path, force: bool = False) -> str:
    dest = dest_dir / name
    if dest.exists() and not force:
        return f"[skip] {name} 已存在"

    bootstrap_env()
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        src = attempt_download_asset(name)
        src_path = Path(src)
        if not src_path.exists():
            from src.core.third_party_paths import import_yolo

            model = import_yolo()(name)
            src_path = Path(getattr(model, "ckpt_path", None) or name)
        shutil.copy2(src_path, dest)
        return f"[ok] {name} -> {dest}"
    except Exception as e:
        return f"[fail] {name}: {e}"


def main():
    parser = argparse.ArgumentParser(description="下载 YOLO 预训练权重到 weights/pretrained/")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    parser.add_argument("--names", nargs="*", default=None, help="指定权重名，如 yolov8s.pt")
    args = parser.parse_args()

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    names = args.names or list_pretrained_names()

    print(f"目标目录: {PRETRAINED_DIR}")
    print(f"共 {len(names)} 个权重\n")

    ok, fail = 0, 0
    for name in names:
        msg = download_one(name, PRETRAINED_DIR, force=args.force)
        print(msg)
        if msg.startswith("[ok]") or msg.startswith("[skip]"):
            ok += 1
        else:
            fail += 1

    print(f"\n完成: 成功/跳过 {ok}, 失败 {fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
