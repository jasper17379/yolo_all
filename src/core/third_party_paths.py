"""
third_party 目录统一路径：源码仓库 + 业务模型权重。

目标：除 pip 通用库外，业务源码与模型缓存均落在项目内，便于离线移植。
"""

from __future__ import annotations

import os
import shutil
import sys
import types
from pathlib import Path

from src.core.config import PROJECT_ROOT

THIRD_PARTY = PROJECT_ROOT / "third_party"
MODELS_ROOT = THIRD_PARTY / "models"
INSIGHTFACE_ROOT = MODELS_ROOT / "insightface"
PADDLEOCR_ROOT = MODELS_ROOT / "paddleocr"
INSIGHTFACE_REPO = THIRD_PARTY / "insightface"
INSIGHTFACE_PYTHON = INSIGHTFACE_REPO / "python-package"
HYPERLPR_REPO = THIRD_PARTY / "HyperLPR"
HYPERLPR_PYTHON = HYPERLPR_REPO / "Prj-Python"
HYPERLPR_MODELS_ROOT = MODELS_ROOT / "hyperlpr3"
ULTRALYTICS_ROOT = THIRD_PARTY / "ultralytics"
WEIGHTS_PRETRAINED = PROJECT_ROOT / "weights" / "pretrained"

_VENDOR_PATHS_BOOTSTRAPPED = False
_HYPERLPR_SETTINGS_PATCHED = False


def ensure_third_party_dirs() -> None:
    for d in (
        THIRD_PARTY,
        MODELS_ROOT,
        INSIGHTFACE_ROOT,
        INSIGHTFACE_ROOT / "models",
        PADDLEOCR_ROOT,
        HYPERLPR_MODELS_ROOT,
        THIRD_PARTY / "ultralytics_config",
        WEIGHTS_PRETRAINED,
    ):
        d.mkdir(parents=True, exist_ok=True)


def insightface_model_dir(model_name: str = "buffalo_l") -> Path:
    return INSIGHTFACE_ROOT / "models" / model_name


def migrate_insightface_from_user_home(model_name: str = "buffalo_l") -> Path | None:
    src = Path.home() / ".insightface" / "models" / model_name
    dst = insightface_model_dir(model_name)
    if not src.exists() or not any(src.iterdir()):
        return None
    if dst.exists() and any(dst.iterdir()):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[migrate] InsightFace: {src} -> {dst}")
    return dst


def migrate_paddleocr_from_user_home() -> Path | None:
    """将 ~/.paddlex 缓存迁移到 third_party/models/paddleocr/。"""
    src = Path.home() / ".paddlex"
    dst = PADDLEOCR_ROOT
    if not src.exists() or not any(src.iterdir()):
        return dst if dst.exists() and any(dst.iterdir()) else None
    if dst.exists() and any(dst.iterdir()):
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[migrate] PaddleOCR/PaddleX: {src} -> {dst}")
    return dst


def migrate_hyperlpr3_from_user_home() -> Path | None:
    """将 ~/.hyperlpr3 迁移到 third_party/models/hyperlpr3/。"""
    candidates = [
        Path.home() / ".hyperlpr3",
        Path(os.environ.get("USERPROFILE", "")) / ".hyperlpr3",
    ]
    homedrive = os.environ.get("HOMEDRIVE", "")
    homepath = os.environ.get("HOMEPATH", "")
    if homedrive and homepath:
        candidates.append(Path(homedrive + homepath) / ".hyperlpr3")
    dst = HYPERLPR_MODELS_ROOT
    for src in candidates:
        if not src.exists() or not any(src.iterdir()):
            continue
        if dst.exists() and any(dst.iterdir()):
            return dst
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"[migrate] HyperLPR3: {src} -> {dst}")
        return dst
    return dst if dst.exists() and any(dst.iterdir()) else None


def _inject_hyperlpr3_settings() -> bool:
    """
    在 import hyperlpr3 之前注入 settings，将模型目录指向项目内。
    避免上游 Windows HOMEPATH 路径错误及 import 时自动下载到用户目录。
    """
    if not (HYPERLPR_PYTHON / "hyperlpr3" / "config" / "settings.py").exists():
        return False
    import importlib.util

    HYPERLPR_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    home = str(HYPERLPR_MODELS_ROOT.resolve())
    os.environ["HYPERLPR3_HOME"] = home

    if "hyperlpr3.config.settings" in sys.modules:
        sys.modules["hyperlpr3.config.settings"]._DEFAULT_FOLDER_ = home
        return True

    settings_path = HYPERLPR_PYTHON / "hyperlpr3" / "config" / "settings.py"
    spec = importlib.util.spec_from_file_location("hyperlpr3.config.settings", settings_path)
    if spec is None or spec.loader is None:
        return False
    settings_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings_mod)
    settings_mod._DEFAULT_FOLDER_ = home

    config_pkg = types.ModuleType("hyperlpr3.config")
    config_pkg.__path__ = [str(HYPERLPR_PYTHON / "hyperlpr3" / "config")]
    config_pkg.__package__ = "hyperlpr3.config"
    config_pkg.settings = settings_mod
    sys.modules["hyperlpr3.config"] = config_pkg
    sys.modules["hyperlpr3.config.settings"] = settings_mod
    # 不注册 hyperlpr3 根模块，避免挡住后续真实包加载
    return True


def setup_vendor_python_paths() -> None:
    """将 vendored 源码目录加入 sys.path（优先于 site-packages）。"""
    global _VENDOR_PATHS_BOOTSTRAPPED
    if _VENDOR_PATHS_BOOTSTRAPPED:
        return
    # 仓库根目录下含 ultralytics/、hyperlpr3/ 等 Python 包
    vendors = [
        ULTRALYTICS_ROOT,
        HYPERLPR_PYTHON,
        INSIGHTFACE_PYTHON,
    ]
    for p in vendors:
        ps = str(p.resolve())
        if p.exists() and ps not in sys.path:
            sys.path.insert(0, ps)
    _VENDOR_PATHS_BOOTSTRAPPED = True


def insightface_available() -> bool:
    """third_party/insightface/python-package 是否可 import。"""
    bootstrap_env()
    pkg = INSIGHTFACE_PYTHON / "insightface" / "__init__.py"
    if not pkg.exists():
        return False
    try:
        from insightface.app import FaceAnalysis  # noqa: F401

        return True
    except Exception:
        return False


def import_face_analysis():
    """从 third_party/insightface 加载 FaceAnalysis。"""
    bootstrap_env()
    if not (INSIGHTFACE_PYTHON / "insightface").exists():
        raise RuntimeError(
            "InsightFace 源码缺失。请运行: python scripts/setup_third_party.py"
        )
    from insightface.app import FaceAnalysis

    return FaceAnalysis


def ultralytics_available() -> bool:
    """third_party/ultralytics 是否可 import。"""
    bootstrap_env()
    pkg = ULTRALYTICS_ROOT / "ultralytics" / "__init__.py"
    if not pkg.exists():
        return False
    try:
        import ultralytics  # noqa: F401

        return hasattr(ultralytics, "YOLO")
    except Exception:
        return False


def import_yolo():
    """从 third_party/ultralytics 加载 YOLO（bootstrap 后 import）。"""
    bootstrap_env()
    if not (ULTRALYTICS_ROOT / "ultralytics").exists():
        raise RuntimeError(
            "Ultralytics 源码缺失。请运行: python scripts/setup_yolo_sources.py"
        )
    from ultralytics import YOLO

    return YOLO


def patch_hyperlpr3_settings() -> bool:
    """将 HyperLPR3 模型目录指向项目内。"""
    global _HYPERLPR_SETTINGS_PATCHED
    if _HYPERLPR_SETTINGS_PATCHED:
        return True
    if not (HYPERLPR_PYTHON / "hyperlpr3").exists():
        return False
    setup_vendor_python_paths()
    ok = _inject_hyperlpr3_settings()
    _HYPERLPR_SETTINGS_PATCHED = ok
    return ok


def prepare_hyperlpr3() -> bool:
    """准备 HyperLPR3 源码与模型路径，返回是否可用。"""
    bootstrap_env()
    if not (HYPERLPR_PYTHON / "hyperlpr3").exists():
        return False
    return patch_hyperlpr3_settings()


def hyperlpr3_available() -> bool:
    if not patch_hyperlpr3_settings():
        return False
    try:
        import hyperlpr3 as lpr3

        return hasattr(lpr3, "LicensePlateCatcher")
    except Exception:
        return False


def download_hyperlpr3_models(re_download: bool = False) -> bool:
    """下载 HyperLPR3 ONNX 模型到 third_party/models/hyperlpr3/。"""
    import tempfile
    import zipfile

    import requests
    from tqdm import tqdm

    if not (HYPERLPR_PYTHON / "hyperlpr3").exists():
        print("[fail] HyperLPR 源码缺失: third_party/HyperLPR/Prj-Python")
        return False

    version = "20230229"
    models_dir = HYPERLPR_MODELS_ROOT / version
    if models_dir.exists() and any(models_dir.rglob("*.onnx")) and not re_download:
        print(f"[skip] HyperLPR3 模型已存在: {models_dir}")
        return True

    HYPERLPR_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    url = f"http://hyperlpr.tunm.top/raw/{version}.zip"
    print(f"[download] HyperLPR3: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with tqdm(total=total, unit="iB", unit_scale=True, desc="HyperLPR3") as bar:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        tmp.write(chunk)
                        bar.update(len(chunk))
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(HYPERLPR_MODELS_ROOT)
        tmp_path.unlink(missing_ok=True)
        print(f"[ok] HyperLPR3 模型目录: {models_dir}")
        return models_dir.exists()
    except Exception as e:
        print(f"[fail] HyperLPR3 模型下载失败: {e}")
        # 回退调用上游 initialization（settings 已注入时可用）
        if patch_hyperlpr3_settings():
            try:
                from hyperlpr3.config.configuration import initialization

                initialization(re_download=re_download)
                return models_dir.exists()
            except Exception as e2:
                print(f"[fail] HyperLPR3 回退下载也失败: {e2}")
        return False


def warmup_paddleocr_models(device: str = "cpu") -> bool:
    """触发 PaddleOCR 下载模型到 PADDLE_PDX_CACHE_HOME。"""
    bootstrap_env()
    try:
        from paddleocr import PaddleOCR

        use_gpu = device not in ("cpu",)
        try:
            PaddleOCR(use_angle_cls=True, lang="ch", device="gpu:0" if use_gpu else "cpu")
        except TypeError:
            PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu)
        print(f"[ok] PaddleOCR 模型缓存: {PADDLEOCR_ROOT}")
        return True
    except Exception as e:
        print(f"[fail] PaddleOCR 预热失败: {e}")
        return False


def bootstrap_env() -> None:
    ensure_third_party_dirs()
    migrate_insightface_from_user_home()
    migrate_paddleocr_from_user_home()
    migrate_hyperlpr3_from_user_home()
    setup_vendor_python_paths()
    patch_hyperlpr3_settings()

    os.environ.setdefault("INSIGHTFACE_HOME", str(INSIGHTFACE_ROOT))
    # PaddleOCR 3.x / PaddleX 实际读取 PADDLE_PDX_CACHE_HOME
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLEOCR_ROOT))
    os.environ.setdefault("PADDLE_PDX_HOME", str(PADDLEOCR_ROOT))
    os.environ.setdefault("PADDLEX_HOME", str(PADDLEOCR_ROOT))
    os.environ.setdefault("PADDLEOCR_HOME", str(PADDLEOCR_ROOT))
    os.environ.setdefault("HYPERLPR3_HOME", str(HYPERLPR_MODELS_ROOT))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "third_party" / "ultralytics_config"))


def vendor_status() -> dict[str, dict]:
    """检查各业务依赖是否已在项目内就绪（供离线移植核对）。"""
    bootstrap_env()
    return {
        "ultralytics": {
            "source": str(ULTRALYTICS_ROOT),
            "source_ok": (ULTRALYTICS_ROOT / "ultralytics").exists(),
            "import_ok": ultralytics_available(),
        },
        "hyperlpr3": {
            "source": str(HYPERLPR_PYTHON),
            "source_ok": (HYPERLPR_PYTHON / "hyperlpr3").exists(),
            "models": str(HYPERLPR_MODELS_ROOT),
            "models_ok": HYPERLPR_MODELS_ROOT.exists() and any(HYPERLPR_MODELS_ROOT.iterdir()),
            "import_ok": hyperlpr3_available(),
        },
        "insightface": {
            "source": str(INSIGHTFACE_PYTHON),
            "source_ok": (INSIGHTFACE_PYTHON / "insightface").exists(),
            "models": str(insightface_model_dir()),
            "models_ok": insightface_model_dir().exists() and any(insightface_model_dir().iterdir()),
            "import_ok": insightface_available(),
        },
        "paddleocr": {
            "models": str(PADDLEOCR_ROOT),
            "models_ok": PADDLEOCR_ROOT.exists() and any(PADDLEOCR_ROOT.iterdir()),
        },
        "yolo_pretrained": {
            "path": str(WEIGHTS_PRETRAINED),
            "models_ok": WEIGHTS_PRETRAINED.exists() and any(WEIGHTS_PRETRAINED.glob("*.pt")),
        },
    }
