#!/usr/bin/env python3
"""列出当前环境可用计算设备。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.device import cuda_available, device_label, gpu_count, resolve_yolo_device


def main():
    print(f"CUDA 可用: {cuda_available()}")
    print(f"GPU 数量: {gpu_count()}")
    for name in ("auto", "cpu", "0", "cuda:0"):
        try:
            print(f"  --device {name:8} -> {device_label(name)} (yolo={resolve_yolo_device(name)})")
        except RuntimeError as e:
            print(f"  --device {name:8} -> 错误: {e}")


if __name__ == "__main__":
    main()
