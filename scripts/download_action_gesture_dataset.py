"""
从 Bing / 百度图库下载真实人物动作/手势图片，YOLO 自动标注人体框。

质量要求（默认）：
  - 必须检测到 person（COCO），且人体框占图面积 >= 8%
  - 拒绝文字/Logo/图标类图片（OK 字母图等）
  - fighting 优先保留检测到 >=2 人的图
  - 手势类要求人物可见（含人脸/上半身），非纯手部特写

用法:
  python scripts/download_action_gesture_dataset.py --clean-all --per-class 15
  python scripts/download_action_gesture_dataset.py --classes ok,peace,fighting --replace-classes --per-class 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.third_party_paths import bootstrap_env

bootstrap_env()

DATASETS = PROJECT_ROOT / "datasets" / "action"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# class_id -> (name, bing_queries, baidu_queries, min_persons)
ACTION_SPECS: dict[int, tuple[str, list[str], list[str], int]] = {
    0: (
        "normal",
        ["person standing full body real photo portrait", "adult standing outdoor real photo"],
        ["真人 站立 全身 照片", "行人 站立 户外 照片"],
        1,
    ),
    1: (
        "smoking",
        ["man smoking cigarette real photo face visible", "person smoking outdoor portrait photo"],
        ["男人 抽烟 真人 照片", "吸烟 人物 面部 照片"],
        1,
    ),
    2: (
        "fighting",
        [
            "two people fighting punch real photo street",
            "men fighting each other real photo not cartoon",
            "people brawl fight real photograph",
        ],
        ["两人 打架 真人 照片", "街头 打架 人物 真实照片", "互相 殴打 真人"],
        2,
    ),
    3: (
        "falling",
        [
            "person falling down floor real photo accident",
            "man slipped and fell ground real photograph",
            "elderly person fall down real photo help",
        ],
        ["人 摔倒 地上 真人 照片", "跌倒 事故 人物 真实照片", "滑倒 摔倒 真人"],
        1,
    ),
    4: (
        "ok",
        [
            "person making OK hand sign face visible real photo",
            "woman OK gesture selfie real photo human",
            "man OK hand sign portrait photo not logo",
        ],
        ["真人 比OK 手势 面部 照片", "OK手势 自拍 人物 照片", "人手 比OK 人脸 可见"],
        1,
    ),
    5: (
        "middle_finger",
        [
            "person flipping middle finger real photo face",
            "man giving middle finger gesture portrait photo",
            "woman middle finger sign real photograph human",
        ],
        ["竖中指 真人 面部 照片", "比中指 手势 人物 照片", "对人 竖中指 自拍"],
        1,
    ),
    6: (
        "thumbs_up",
        [
            "person thumbs up gesture face visible real photo",
            "man thumb up sign portrait real photograph",
            "woman thumbs up selfie real photo",
        ],
        ["竖大拇指 真人 面部 照片", "点赞 手势 人物 自拍", "拇指 向上 真人 照片"],
        1,
    ),
    7: (
        "peace",
        [
            "person peace sign V gesture face visible real photo",
            "woman victory hand sign selfie real photograph",
            "young man peace fingers portrait photo human",
        ],
        ["剪刀手 比耶 真人 面部 照片", "V字 手势 自拍 人物", "比耶 手势 人像 照片"],
        1,
    ),
    8: (
        "fist",
        [
            "person raised fist gesture face visible real photo",
            "man clenched fist portrait photograph human",
            "protest fist pump person real photo face",
        ],
        ["握拳 手势 真人 面部 照片", "拳头 举起 人物 照片", "握拳 人像 照片"],
        1,
    ),
}

NAME_TO_ID = {v[0]: k for k, v in ACTION_SPECS.items()}
MIN_PERSON_AREA = 0.08  # 人体框至少占图 8%


def _img_hash(img: np.ndarray) -> str:
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


def _img_dhash(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = gray[:, 1:] > gray[:, :-1]
    bits = diff.flatten()
    return sum((1 << i) for i, v in enumerate(bits) if v)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _is_near_duplicate(dh: int, seen: set[int], max_dist: int = 10) -> bool:
    return any(_hamming(dh, s) <= max_dist for s in seen)


def _load_used_hashes(exclude_class_id: int | None = None) -> tuple[set[str], set[int]]:
    """已入库图片 md5 与 dhash（replace 时可排除当前类）。"""
    md5s: set[str] = set()
    dhashes: set[int] = set()
    for split in ("train", "val"):
        lbl_dir = DATASETS / "labels" / split
        img_dir = DATASETS / "images" / split
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            with open(lbl, encoding="utf-8") as f:
                line = f.readline().strip()
            if not line:
                continue
            cid = int(line.split()[0])
            if exclude_class_id is not None and cid == exclude_class_id:
                continue
            img_path = img_dir / f"{lbl.stem}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is not None:
                md5s.add(_img_hash(img))
                dhashes.add(_img_dhash(img))
    return md5s, dhashes


def _load_used_urls() -> set[str]:
    p = DATASETS / "reference" / "used_urls.json"
    if p.exists():
        return set(json.loads(p.read_text(encoding="utf-8")))
    return set()


def _save_used_urls(urls: set[str]) -> None:
    ref = DATASETS / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "used_urls.json").write_text(json.dumps(sorted(urls), ensure_ascii=False, indent=2), encoding="utf-8")


def _search_bing(queries: list[str], limit: int) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if len(urls) >= limit:
            break
        q_full = f"{q} -logo -icon -clipart -illustration -vector -emoji -button"
        for first in range(0, min(limit * 5, 175), 35):
            if len(urls) >= limit:
                break
            try:
                r = SESSION.get(
                    "https://www.bing.com/images/async",
                    params={"q": q_full, "first": first, "count": 35, "relp": 35, "qft": "+filterui:photo-photo"},
                    headers={"Referer": "https://www.bing.com/images/"},
                    timeout=25,
                )
                for u in re.findall(r'murl&quot;:&quot;(.*?)&quot;', r.text):
                    u = u.replace("\\u0026", "&")
                    if u.startswith("http") and u not in seen:
                        seen.add(u)
                        urls.append(u)
                        if len(urls) >= limit:
                            break
            except Exception as e:
                print(f"  [Bing] {q[:28]}... first={first}: {e}")
            time.sleep(0.2)
    return urls


def _search_baidu(queries: list[str], limit: int) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if len(urls) >= limit:
            break
        ref = "https://image.baidu.com/search/index?tn=baiduimage&word=" + requests.utils.quote(q)
        for pn in range(0, min(limit * 4, 120), 30):
            if len(urls) >= limit:
                break
            try:
                r = SESSION.get(
                    "https://image.baidu.com/search/acjson",
                    params={
                        "tn": "resultjson_com",
                        "ipn": "rj",
                        "ct": "201326592",
                        "fp": "result",
                        "word": q,
                        "queryWord": q,
                        "pn": pn,
                        "rn": 30,
                        "ie": "utf-8",
                    },
                    headers={"Referer": ref, "Accept": "application/json, text/plain, */*"},
                    timeout=20,
                )
                text = r.text.replace("\\'", "'")
                try:
                    items = json.loads(text).get("data") or []
                    for it in items:
                        if not it:
                            continue
                        for key in ("middleURL", "thumbURL", "hoverURL", "objURL"):
                            u = it.get(key) or ""
                            if u.startswith("http") and u not in seen:
                                seen.add(u)
                                urls.append(u)
                                break
                except json.JSONDecodeError:
                    for m in re.finditer(r'"(?:middleURL|thumbURL|hoverURL|objURL)"\s*:\s*"([^"]+)"', text):
                        u = m.group(1).replace("\\/", "/")
                        if u.startswith("http") and u not in seen:
                            seen.add(u)
                            urls.append(u)
            except Exception as e:
                print(f"  [百度] {q[:18]}... pn={pn}: {e}")
            time.sleep(0.3)
    return urls


def _download_image(url: str) -> np.ndarray | None:
    url = url.replace("\\/", "/")
    for ref in ("https://www.bing.com/", "https://image.baidu.com/", None):
        try:
            headers = {"User-Agent": UA}
            if ref:
                headers["Referer"] = ref
            resp = SESSION.get(url, headers=headers, timeout=30, allow_redirects=True)
            if len(resp.content) < 8000:
                continue
            ct = (resp.headers.get("content-type") or "").lower()
            if "text" in ct or "html" in ct or "svg" in ct:
                continue
            img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if min(h, w) < 200 or h * w < 200 * 200:
                continue
            return img
        except Exception:
            continue
    return None


def _is_text_or_logo(img: np.ndarray) -> bool:
    """拒绝 OK 字母、Logo、纯图标等非照片。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 颜色丰富度：扁平 graphic 颜色种类少
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    quantized = (small // 32).reshape(-1, 3)
    unique = len({tuple(c) for c in quantized})
    if unique < 12:
        return True
    # 大面积纯色背景 + 中央高对比边缘（像文字牌）
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(np.mean(bw > 200))
    black_ratio = float(np.mean(bw < 55))
    if white_ratio > 0.72 or black_ratio > 0.72:
        edges = cv2.Canny(gray, 80, 180)
        edge_ratio = float(np.mean(edges > 0))
        if edge_ratio < 0.04 or edge_ratio > 0.35:
            return True
    # 长宽比极端且边缘少 → 横幅/文字条
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 3.2 and float(np.mean(cv2.Canny(gray, 50, 150) > 0)) < 0.06:
        return True
    # 饱和度过低且亮度集中 → 灰度 logo
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))
    if sat_mean < 18 and val_std < 28:
        return True
    return False


def _get_person_detector():
    from src.core.yolo_wrapper import YOLOWrapper

    return YOLOWrapper(version="yolov8", weights="yolov8n.pt", model_size="n")


def _detect_persons(img: np.ndarray, detector) -> list[tuple[float, float, float, float, float]]:
    """返回 [(cx,cy,bw,bh,area_ratio), ...] 按面积降序。"""
    h, w = img.shape[:2]
    img_area = h * w
    out: list[tuple[float, float, float, float, float]] = []
    try:
        results = detector.predict(source=img, conf=0.25, imgsz=640, device="cpu", verbose=False, save=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area_ratio = (x2 - x1) * (y2 - y1) / img_area
                if area_ratio < MIN_PERSON_AREA * 0.5:
                    continue
                out.append(
                    (
                        (x1 + x2) / 2 / w,
                        (y1 + y2) / 2 / h,
                        (x2 - x1) / w,
                        (y2 - y1) / h,
                        area_ratio,
                    )
                )
    except Exception:
        pass
    out.sort(key=lambda x: x[4], reverse=True)
    return out


def _pick_bbox(
    persons: list[tuple[float, float, float, float, float]],
    min_persons: int,
    min_area: float = MIN_PERSON_AREA,
) -> tuple[float, float, float, float] | None:
    if len(persons) < min_persons:
        return None
    cx, cy, bw, bh, area = persons[0]
    if area < min_area:
        return None
    return cx, cy, bw, bh


def _validate_image(
    img: np.ndarray,
    detector,
    class_id: int,
    min_persons: int,
) -> tuple[float, float, float, float] | None:
    if _is_text_or_logo(img):
        return None
    persons = _detect_persons(img, detector)
    min_area = MIN_PERSON_AREA
    if class_id in (4, 5, 6, 7, 8):
        # 手势：人体需可见，框不可过小
        min_area = max(0.06, MIN_PERSON_AREA * 0.75)
    return _pick_bbox(persons, min_persons=min_persons, min_area=min_area)


def _write_yolo(path: Path, class_id: int, cx: float, cy: float, w: float, h: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def _save_sample(split: str, name: str, img: np.ndarray, class_id: int, bbox: tuple[float, float, float, float]) -> None:
    cx, cy, w, h = bbox
    img_path = DATASETS / "images" / split / f"{name}.jpg"
    lbl_path = DATASETS / "labels" / split / f"{name}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    _write_yolo(lbl_path, class_id, cx, cy, w, h)


def _count_split(class_id: int) -> tuple[int, int]:
    train_n = val_n = 0
    for split in ("train", "val"):
        lbl_dir = DATASETS / "labels" / split
        if not lbl_dir.exists():
            continue
        for p in lbl_dir.glob("*.txt"):
            with open(p, encoding="utf-8") as f:
                line = f.readline().strip()
            if line and int(line.split()[0]) == class_id:
                if split == "train":
                    train_n += 1
                else:
                    val_n += 1
    return train_n, val_n


def _remove_class_images(class_id: int) -> int:
    name = ACTION_SPECS[class_id][0]
    n = 0
    for split in ("train", "val"):
        img_dir = DATASETS / "images" / split
        if not img_dir.exists():
            continue
        for img in list(img_dir.glob(f"action_real_{name}_*.jpg")):
            lbl = DATASETS / "labels" / split / f"{img.stem}.txt"
            img.unlink(missing_ok=True)
            if lbl.exists():
                lbl.unlink()
            n += 1
    print(f"  已删除 {name} 旧图 {n} 张")
    return n


def _clean_all() -> None:
    for sub in ("images", "labels"):
        p = DATASETS / sub
        if p.exists():
            shutil.rmtree(p)
    print("已清空 datasets/action/images 与 labels")


def _collect_urls(bing_q: list[str], baidu_q: list[str], need: int, sources: list[str]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    if "bing" in sources:
        urls.extend(_search_bing(bing_q, need * 6))
    if "baidu" in sources:
        urls.extend(_search_baidu(baidu_q, need * 6))
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def download_class(
    class_id: int,
    per_class: int,
    val_ratio: float,
    detector,
    sources: list[str],
    replace: bool,
    used_md5: set[str],
    used_dhash: set[int],
    used_urls: set[str],
) -> int:
    name, bing_q, baidu_q, min_persons = ACTION_SPECS[class_id]
    if replace:
        _remove_class_images(class_id)
        used_md5.clear()
        used_dhash.clear()
        m, d = _load_used_hashes(exclude_class_id=class_id)
        used_md5.update(m)
        used_dhash.update(d)

    exist = sum(_count_split(class_id))
    need = max(0, per_class - exist)
    if need == 0:
        print(f"[{name}] 已有 {exist} 张，跳过")
        return 0

    val_target = max(1, int(per_class * val_ratio))
    urls = _collect_urls(bing_q, baidu_q, need + 80, sources)
    print(f"  [{name}] 候选 URL {len(urls)}，需 {need} 张（去重 md5={len(used_md5)} url={len(used_urls)}）")

    saved = 0
    rejected = {"dl": 0, "text": 0, "person": 0, "dup": 0}
    for url in urls:
        if saved >= need:
            break
        norm_url = url.split("?")[0]
        if norm_url in used_urls:
            rejected["dup"] += 1
            continue
        img = _download_image(url)
        if img is None:
            rejected["dl"] += 1
            continue
        ih = _img_hash(img)
        dh = _img_dhash(img)
        if ih in used_md5 or _is_near_duplicate(dh, used_dhash):
            rejected["dup"] += 1
            continue
        if _is_text_or_logo(img):
            rejected["text"] += 1
            continue
        bbox = _validate_image(img, detector, class_id, min_persons)
        if bbox is None:
            rejected["person"] += 1
            continue

        n_train, n_val = _count_split(class_id)
        split = "val" if n_val < val_target else "train"
        idx = exist + saved + 1
        fname = f"action_real_{name}_{idx:03d}"
        _save_sample(split, fname, img, class_id, bbox)
        used_md5.add(ih)
        used_dhash.add(dh)
        used_urls.add(norm_url)
        saved += 1
        print(f"  [ok] {fname} ({split}) {img.shape[1]}x{img.shape[0]}")

    if saved < need:
        print(
            f"  [warn] {name} 仅 {saved}/{need} 张 | "
            f"拒绝: 下载{rejected['dl']} logo{rejected['text']} 无人体{rejected['person']} 重复{rejected['dup']}"
        )
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="下载真实人物动作/手势图片")
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--sources", default="bing,baidu")
    parser.add_argument("--classes", default="", help="仅处理这些类，逗号分隔，如 ok,peace,fighting")
    parser.add_argument("--replace-classes", action="store_true", help="先删指定类旧图再重下")
    parser.add_argument("--clean-all", action="store_true")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.clean_all:
        _clean_all()

    if args.classes.strip():
        class_ids = []
        for part in args.classes.split(","):
            part = part.strip()
            if part.isdigit():
                class_ids.append(int(part))
            elif part in NAME_TO_ID:
                class_ids.append(NAME_TO_ID[part])
            else:
                raise SystemExit(f"未知类别: {part}")
    else:
        class_ids = sorted(ACTION_SPECS.keys())

    print("==> 加载 YOLO 人体检测器...")
    detector = _get_person_detector()
    used_md5, used_dhash = _load_used_hashes()
    used_urls = _load_used_urls()

    total = 0
    for class_id in class_ids:
        cname = ACTION_SPECS[class_id][0]
        exist = sum(_count_split(class_id))
        print(f"\n==> 类别 {class_id} {cname}（当前 {exist}，目标 {args.per_class}）")
        total += download_class(
            class_id,
            args.per_class,
            args.val_ratio,
            detector,
            sources,
            args.replace_classes,
            used_md5,
            used_dhash,
            used_urls,
        )

    _save_used_urls(used_urls)

    print("\n==> 数据集统计:")
    for split in ("train", "val"):
        imgs = list((DATASETS / "images" / split).glob("*.jpg"))
        print(f"  {split}: {len(imgs)} 张")
    for cid in class_ids:
        n = sum(_count_split(cid))
        print(f"    {ACTION_SPECS[cid][0]}: {n}")

    print(f"\n完成，新增 {total} 张。")
    print("训练: python -m src.train.trainer --task action --epochs 10 --batch 4 --device 0")


if __name__ == "__main__":
    main()
