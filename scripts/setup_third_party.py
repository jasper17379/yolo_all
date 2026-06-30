#!/usr/bin/env python3
"""
拉取业务场景依赖源码 + 整理模型到 third_party/models/。

用法:
  python scripts/setup_third_party.py              # 全部
  python scripts/setup_third_party.py --skip-clone   # 仅迁移已有模型
  python scripts/setup_third_party.py --download-face-model
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.third_party_paths import (
    INSIGHTFACE_ROOT,
    PADDLEOCR_ROOT,
    THIRD_PARTY,
    ensure_third_party_dirs,
    insightface_model_dir,
    migrate_insightface_from_user_home,
)

REPOS = {
    "insightface": {
        "url": "https://github.com/deepinsight/insightface.git",
        "branch": "master",
        "desc": "人脸识别 InsightFace",
    },
    "PaddleOCR": {
        "url": "https://github.com/PaddlePaddle/PaddleOCR.git",
        "branch": "main",
        "desc": "车牌 OCR PaddleOCR",
    },
}


def clone_repo(name: str, force: bool = False) -> bool:
    cfg = REPOS[name]
    dest = THIRD_PARTY / name
    if dest.exists() and (dest / ".git").exists():
        if not force:
            print(f"[skip] {name} 已存在")
            return True
        shutil.rmtree(dest)
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", "--branch", cfg["branch"], cfg["url"], str(dest)]
    print(f"[clone] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode == 0


def download_insightface_model(model_name: str = "buffalo_l") -> Path | None:
    dst = insightface_model_dir(model_name)
    if dst.exists() and any(dst.iterdir()):
        print(f"[skip] InsightFace 模型已存在: {dst}")
        return dst

    migrated = migrate_insightface_from_user_home(model_name)
    if migrated and any(migrated.iterdir()):
        print(f"[ok] 已从用户目录迁移: {migrated}")
        return migrated

    try:
        from insightface.utils import ensure_available

        ensure_available("models", model_name, root=str(INSIGHTFACE_ROOT))
        if dst.exists():
            print(f"[ok] 已下载 InsightFace 模型: {dst}")
            return dst
    except Exception as e:
        print(f"[warn] insightface 自动下载失败: {e}")

    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=model_name, root=str(INSIGHTFACE_ROOT), providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0)
        if dst.exists():
            print(f"[ok] FaceAnalysis 触发下载: {dst}")
            return dst
    except Exception as e:
        print(f"[fail] 无法下载 {model_name}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="整理 third_party 源码与模型")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-face-model", action="store_true")
    parser.add_argument("--model", default="buffalo_l")
    args = parser.parse_args()

    ensure_third_party_dirs()
    print(f"模型目录:\n  InsightFace: {INSIGHTFACE_ROOT}\n  PaddleOCR:   {PADDLEOCR_ROOT}\n")

    if not args.skip_clone:
        for name in REPOS:
            clone_repo(name, force=args.force)

    migrate_insightface_from_user_home(args.model)

    if args.download_face_model or not insightface_model_dir(args.model).exists():
        download_insightface_model(args.model)

    # YOLO 源码
    yolo_script = PROJECT_ROOT / "scripts" / "setup_yolo_sources.py"
    if yolo_script.exists() and not args.skip_clone:
        subprocess.run([sys.executable, str(yolo_script)], cwd=PROJECT_ROOT)

    print("\n完成。业务库仍通过 pip 调用，源码在 third_party/ 供阅读移植；")
    print("模型权重在 third_party/models/，不再依赖 C:\\Python312 或用户家目录缓存。")


if __name__ == "__main__":
    main()
