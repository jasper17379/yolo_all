"""
设备选择：CPU / 单卡 GPU / 多卡 GPU / 自动。

Ultralytics YOLO 与 InsightFace ONNX 共用同一套 --device 语义。
"""

from __future__ import annotations

from typing import Any


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def resolve_yolo_device(device: str | None = "auto") -> str | int:
    """
    解析为 Ultralytics 的 device 参数。

    支持:
      auto     - 有 GPU 用 0，否则 cpu
      cpu      - 强制 CPU
      cuda     - 默认 GPU 0
      cuda:0   - 指定 GPU
      0 / 1    - GPU 编号
      0,1      - 多卡训练
    """
    d = (device or "auto").strip().lower()
    if d == "auto":
        return 0 if cuda_available() else "cpu"
    if d == "cpu":
        return "cpu"
    if d == "cuda":
        if not cuda_available():
            raise RuntimeError("指定了 GPU 但未检测到 CUDA，请改用 --device cpu")
        return 0
    if d.startswith("cuda:"):
        if not cuda_available():
            raise RuntimeError(f"指定了 {d} 但未检测到 CUDA")
        return d
    if "," in d:
        if not cuda_available():
            raise RuntimeError(f"多卡 {d} 需要 CUDA")
        return d
    if d.isdigit():
        if not cuda_available():
            raise RuntimeError(f"GPU {d} 不可用")
        return int(d)
    return d


def device_label(device: str | None = "auto") -> str:
    """人类可读设备描述，用于日志。"""
    resolved = resolve_yolo_device(device)
    if resolved == "cpu":
        return "cpu"
    if isinstance(resolved, int):
        return f"cuda:{resolved}"
    return str(resolved)


def resolve_onnx_providers(device: str | None = "auto") -> list[str]:
    """
    InsightFace ONNX Runtime providers。

    auto/cuda/0 → 优先 CUDA，不可用时回退 CPU（「GPU+CPU 回退」）。
    cpu → 仅 CPU。
    """
    d = (device or "auto").strip().lower()
    if d == "cpu":
        return ["CPUExecutionProvider"]
    providers: list[str] = []
    if d in ("auto", "cuda") or d.startswith("cuda") or d.isdigit() or "," in d:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def resolve_face_ctx_id(device: str | None = "auto") -> int:
    """InsightFace prepare(ctx_id): GPU 为 0/1/...，CPU 为 -1。"""
    resolved = resolve_yolo_device(device)
    if resolved == "cpu":
        return -1
    if isinstance(resolved, int):
        return resolved
    if isinstance(resolved, str) and resolved.isdigit():
        return int(resolved)
    if isinstance(resolved, str) and resolved.startswith("cuda:"):
        try:
            return int(resolved.split(":")[1])
        except (IndexError, ValueError):
            return 0
    return 0


def paddle_use_gpu(device: str | None = "auto") -> bool:
    return resolve_yolo_device(device) != "cpu"


def add_device_arg(parser: Any, default: str = "auto") -> None:
    parser.add_argument(
        "--device",
        default=default,
        help="设备: auto|cpu|cuda|cuda:0|0|0,1 (多卡训练用逗号分隔)",
    )


def print_device_info(device: str) -> None:
    n = gpu_count()
    print(f"[device] 请求={device} -> 实际={device_label(device)} | 可见GPU数={n}")
