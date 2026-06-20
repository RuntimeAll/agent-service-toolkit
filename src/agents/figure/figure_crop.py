# -*- coding: utf-8 -*-
"""figure_crop（vendor 自 codeplace-B/qbank-labeler/figure_crop/detect.py）· DocLayout-YOLO 母题切图。

🔴 PRD-C-100 B-mathfig：去 MCP 壳、进程内直 import（detect_and_crop）；模型单例**只加载一次**
   （_get_model 全局，首次裁图时加载，可在 service lifespan 预热）。torch 全栈进 toolkit 进程，
   numpy 2.3 兼容 B0 已实测（detect q16 检出 2 图 conf 0.94/0.95，2.3s）。

职责（27 号手册 ①切图）：对一张题目图，检出「图形(figure)」区域 → 裁出小图 → 从上到下编号。
🔴 只裁 figure，不碰 isolate_formula（行间公式归 Opus 转 LaTeX，不当图裁）；纯文字/纯公式题 → figures=[]。
🔴 图全链 PNG 无损（cv2.imwrite 默认 PNG 无压缩损失，对齐铁律「禁压缩质量」）。

权重定位（FIGURE_CROP_WEIGHTS env 覆盖）：默认指 codeplace-B/qbank-labeler 权重（40MB .pt 不进
   toolkit git，部署走 download_weights.py 或 copy）。env 显式指向部署机权重路径。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
from doclayout_yolo import YOLOv10

# 权重默认路径：vendor 不带 40MB .pt，默认指 qbank-labeler 现成权重（本机可跑）；
# 部署机经 FIGURE_CROP_WEIGHTS env 指向 download_weights.py 拉下的本地权重。
# 🔴 2026-06-20：原始 repo 是 toolkit/src/agents/figure（parents[4]=toolkit 根），但 prod 镜像把
#   src/agents COPY 成扁平 /app/agents/figure（只 3 层父目录）→ parents[4] IndexError 在 import 期就崩，
#   连带 compose 懒加载走 except 降级（症状=切图永远失败）。容器布局兜底到 /app/weights。
#   prod 实际走 FIGURE_CROP_WEIGHTS env（_weights_path 先查它），此默认值仅本机兜底。
try:
    _DEFAULT_WEIGHTS = (
        Path(__file__).resolve().parents[4]  # toolkit/src/agents/figure → toolkit 根 → ...
        / "weights" / "doclayout_yolo_docstructbench_imgsz1024.pt"
    )
except IndexError:
    _DEFAULT_WEIGHTS = Path("/app/weights/doclayout_yolo_docstructbench_imgsz1024.pt")
_QBANK_WEIGHTS = Path(
    r"d:/workplace/book-ai/codeplace-B/qbank-labeler/figure_crop/weights/"
    r"doclayout_yolo_docstructbench_imgsz1024.pt"
)


def _weights_path() -> str:
    env = os.environ.get("FIGURE_CROP_WEIGHTS")
    if env and Path(env).exists():
        return env
    if _DEFAULT_WEIGHTS.exists():
        return str(_DEFAULT_WEIGHTS)
    return str(_QBANK_WEIGHTS)  # 本机兜底（部署须配 FIGURE_CROP_WEIGHTS 或 vendor 权重）


_OUT_DIR = Path(os.environ.get("FIGURE_CROP_OUT", str(Path(__file__).parent / "out")))
_model: Any = None


def get_model() -> Any:
    """模型单例（只加载一次）。service lifespan 可调一次预热，把冷启动移到启动期。"""
    global _model
    if _model is None:
        _model = YOLOv10(_weights_path())
    return _model


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _dedup(figs: list[dict], iou_thr: float = 0.6) -> list[dict]:
    kept: list[dict] = []
    for r in sorted(figs, key=lambda x: -x["conf"]):
        if all(_iou(r["bbox"], k["bbox"]) <= iou_thr for k in kept):
            kept.append(r)
    return kept


def detect_and_crop(
    image_path: str,
    out_dir: str | None = None,
    *,
    conf: float = 0.2,
    imgsz: int = 1024,
) -> dict:
    """检图形 + 裁剪（CPU）。返回 {image, figures:[{idx,bbox,cls,conf,crop_path}], all_regions:[...]}。

    figures 只含 cls == 'figure'（排除 figure_caption / isolate_formula / table）。纯文字/公式题 → []。
    🔴 PNG 无损落盘（cv2.imwrite .png）。
    """
    model = get_model()
    res = model.predict(image_path, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    names = res.names
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"读不到图：{image_path}")

    od = Path(out_dir) if out_dir else _OUT_DIR
    od.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    regions: list[dict] = []
    for b in res.boxes:
        cls_id = int(b.cls[0])
        name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        regions.append({"cls": name, "conf": round(float(b.conf[0]), 3), "bbox": [x1, y1, x2, y2]})

    figs = [r for r in regions if r["cls"] == "figure"]
    figs = _dedup(figs)
    figs.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    out_figs: list[dict] = []
    for i, r in enumerate(figs, 1):
        x1, y1, x2, y2 = r["bbox"]
        crop = img[max(0, y1):y2, max(0, x1):x2]
        cp = od / f"{stem}_fig{i}.png"
        cv2.imwrite(str(cp), crop)  # PNG 无损
        out_figs.append({
            "idx": i, "bbox": r["bbox"], "cls": r["cls"],
            "conf": r["conf"], "crop_path": str(cp),
        })

    return {"image": image_path, "figures": out_figs, "all_regions": regions}
