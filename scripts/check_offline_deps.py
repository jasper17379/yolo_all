#!/usr/bin/env python3
"""核对离线环境业务依赖是否齐全。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.third_party_paths import HYPERLPR_PYTHON, THIRD_PARTY, vendor_status


def main():
    print("=== 离线业务依赖核对 ===\n")
    print("pip 通用库（需离线 wheel）: torch, opencv-python, onnxruntime, paddleocr, fastapi ...")
    print("业务 Python 包（已 vendored，无需 pip）:")
    print("  ultralytics  -> third_party/ultralytics")
    print("  hyperlpr3    -> third_party/HyperLPR/Prj-Python")
    print("  insightface  -> third_party/insightface/python-package\n")
    print("项目内 vendored 源码:")
    for name in ("HyperLPR", "insightface", "PaddleOCR", "yolov5", "ultralytics"):
        p = THIRD_PARTY / name
        mark = "OK" if p.exists() else "缺失"
        print(f"  [{mark}] third_party/{name}/")

    print(f"\nHyperLPR3 Python 包: {HYPERLPR_PYTHON}")
    print(f"  -> {'OK' if (HYPERLPR_PYTHON / 'hyperlpr3').exists() else '缺失，运行 setup_third_party.py'}")
    from src.core.third_party_paths import INSIGHTFACE_PYTHON
    print(f"\nInsightFace Python 包: {INSIGHTFACE_PYTHON}")
    print(f"  -> {'OK' if (INSIGHTFACE_PYTHON / 'insightface').exists() else '缺失，运行 setup_third_party.py'}\n")

    all_ok = True
    for name, st in vendor_status().items():
        ok = all(v for k, v in st.items() if k.endswith("_ok"))
        mark = "OK" if ok else "WARN"
        if not ok:
            all_ok = False
        print(f"[{mark}] {name}")
        for k, v in st.items():
            if k.endswith("_ok"):
                print(f"       {k}: {v}")
            elif k in ("source", "models", "path"):
                print(f"       {k}: {v}")

    print()
    if all_ok:
        print("全部业务模型已就绪，可离线调试。")
    else:
        print("部分依赖未就绪。联网环境执行:")
        print("  python scripts/setup_third_party.py --download-hyperlpr-models --download-paddle-models --download-face-model")
        print("然后将 third_party/models/ 与 weights/ 一并拷贝到离线机。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
