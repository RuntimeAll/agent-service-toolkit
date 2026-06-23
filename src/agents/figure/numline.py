# -*- coding: utf-8 -*-
r"""PRD-A-023 B9 · 数轴专用确定性配图模板（治「变式数轴图 LLM 自由手搓质量差到离谱」根因）。

🔴 思路（用户已用 GeoGebra 实测此组命令效果完美，直接固化为确定性模板，不让 opus 自由发挥）：
   数轴 = 一条水平 Vector（带右箭头）+ 整数刻度小竖线 + 数值文字在轴下 + 已知点实心圆 + 字母在轴上。
   命令模板（MIN/MAX = 范围两端整数，每个已知点 label=L 值=v）：
     axis=Vector((MIN-0.7,0),(MAX+0.7,0))
     ticks=Sequence(Segment((i,-0.12),(i,0.12)),i,MIN,MAX)
     nums=Sequence(Text(i,(i-0.13,-0.55)),i,MIN,MAX)
     p_<L>=(v,0)                         # 实心点（point_size>0 留可见）
     t_<L>=Text("<L>",(v-0.18,0.62))     # 字母在点上方
   渲染参数：axes=false, grid=false, mono=true, auto_labels=false。
   🔴 引擎纵横比校正把数轴拉成高图（上下大留白）→ **必须 PIL 裁留白**（ImageChops.difference 求 bbox
      + 留 ~24px 边），裁后是 ~1738×202 宽扁图。

🔴 LLM 只抽**结构化点值 + 范围**（不画图）：从变式题干抽 {字母:数值}（仅**已知位置**的点，
   未知/待求点如「E 表示的数是__」不画）+ 范围 [MIN,MAX]（含所有点值、留余）。抽取走 opus（全 opus
   不切模型，确定性出图，LLM 只抽点值）。抽取失败 / 检测不到数轴 → 上层回退 LLM 造图（绝不卡死）。

🔴 安全名：点变量名用 p_A/t_A…（别裸用 x/y/e 等 GeoGebra 保留名；字母靠 Text 显示）。
🔴 非整数点值（分数）少见：先支持整数刻度；分数点用单独 Text 标值（按需，边界情况）。
"""
from __future__ import annotations

import logging
import math
import os
import re
import tempfile
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.figure import mathfig_render

logger = logging.getLogger(__name__)

# 数轴题检测关键词（题干含「数轴」= 强信号）。
_NUMLINE_KEYWORDS = ("数轴",)


def is_number_line(stem: str | None, figure_spec: Any = None) -> bool:
    """数轴题检测：题干或配图决策（layout）含「数轴」关键词。
    保守宽松——命中即走数轴模板；抽点失败仍会回退 LLM 造图，故误判代价低。"""
    blob = str(stem or "")
    # figure_spec 可能是 str / dict({"layout":...})
    if isinstance(figure_spec, str):
        blob += "\n" + figure_spec
    elif isinstance(figure_spec, dict):
        blob += "\n" + str(figure_spec.get("layout") or "")
    low = blob.lower()
    return any(kw.lower() in low for kw in _NUMLINE_KEYWORDS)


# ---------------------------------------------------------------------------
# LLM 抽点（只抽结构化点值 + 范围，不画图）
# ---------------------------------------------------------------------------
_EXTRACT_SYSTEM = (
    "你是数学题信息抽取器。任务：从下面这道**数轴题**的题面里，抽出要画在数轴上的「已知位置的点」"
    "（字母 → 数值）以及数轴的取值范围。**只抽取、不画图、不解题、不推算未知点**。\n"
    "🔴 抽取规则（必守）：\n"
    "  ① 只收**题面已明确给出位置（数值）的点**：如「点 M,N 分别表示 -3,9」→ 收 M=-3,N=9。\n"
    "  ② **待求/未知点不收**：如「点 E 表示的数是 __」「求点 P 的坐标」中的 E/P 位置未知，**一律不收**"
    "（画出来=泄题/瞎猜）。只有题面直接给了数值的点才收。\n"
    "  ③ 点的字母用单个大写字母（A-Z）；数值为整数或简单小数（如 -3、0、9、2.5）。\n"
    "  ④ 范围 [min,max]：取一个能**含住所有已收点值**、两端各留一两格余量的整数区间"
    "（如点值在 -3~9，可给 [-5,11] 或 [-4,10]）；min<max，且都为整数。\n"
    "  ⑤ 若题面没有任何可收的已知点，points 给空对象 {}（上层会回退）。\n"
    "🔴 只输出一个 JSON（不要解释、不要 markdown fence）：\n"
    '{"points":{"M":-3,"N":9},"min":-5,"max":11}\n'
)


async def _extract_points(
    *, stem: str, invoke: Any, parse_json: Any, model: str | None,
) -> dict[str, Any] | None:
    """opus 抽 {字母:数值} + 范围。返回 {"points":{...},"min":int,"max":int} 或 None（抽不出/异常）。"""
    messages = [
        SystemMessage(content=_EXTRACT_SYSTEM),
        HumanMessage(content="【数轴题面】\n" + str(stem or "")),
    ]
    try:
        text = await invoke(messages, model=model, max_tokens=1024, temperature=0.0)
    except Exception as e:  # noqa: BLE001 — 抽点失败 → None（上层回退 LLM 造图）
        logger.info("numline _extract_points invoke 失败: %s", str(e)[:120])
        return None
    data = parse_json(text)
    if not isinstance(data, dict):
        return None
    raw_pts = data.get("points")
    if not isinstance(raw_pts, dict) or not raw_pts:
        return None
    # 归一点值：字母键 + float 值（弃非法项）。
    points: dict[str, float] = {}
    for k, v in raw_pts.items():
        key = str(k).strip()
        if not re.fullmatch(r"[A-Za-z]", key):
            continue
        try:
            points[key.upper()] = float(v)
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    # 范围：优先用 LLM 给的；非法/缺失 → 据点值自算（含住所有点 + 留余）。
    vmin = _to_int(data.get("min"))
    vmax = _to_int(data.get("max"))
    pv = list(points.values())
    lo, hi = min(pv), max(pv)
    if vmin is None or vmax is None or vmin >= vmax or vmin > lo or vmax < hi:
        vmin = int(math.floor(lo)) - 2
        vmax = int(math.ceil(hi)) + 2
    return {"points": points, "min": int(vmin), "max": int(vmax)}


def _to_int(v: Any) -> int | None:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 确定性模板 → GeoGebra 命令
# ---------------------------------------------------------------------------
def build_commands(points: dict[str, float], vmin: int, vmax: int) -> tuple[list[str], list[str]]:
    """据点值 + 范围套用户已验证的数轴模板，返回 (commands, vals)。
    🔴 安全名：点用 Pt<L>、字母 Text 用 Tx<L>（**无下划线**）。
       实测：`p_A` 这种带下划线名会被 GeoGebra 解析成「p 带下标 A」→ 渲成箭头状而非实心圆点；
       `PtA` 这种纯字母名才正常渲成干净实心圆点（对齐用户验证的 numline_test1）。
       字母靠独立 Text 显示在轴上方（auto_labels=False，不让引擎重印内部点名）。"""
    cmds: list[str] = [
        f"axis=Vector(({vmin - 0.7},0),({vmax + 0.7},0))",
        f"ticks=Sequence(Segment((i,-0.12),(i,0.12)),i,{vmin},{vmax})",
        f"nums=Sequence(Text(i,(i-0.13,-0.55)),i,{vmin},{vmax})",
    ]
    vals: list[str] = []
    for label in sorted(points.keys()):
        v = points[label]
        # 整数值用整数写（避免 -6.0），分数保留
        vs = str(int(v)) if float(v).is_integer() else str(v)
        pname = f"Pt{label}"
        cmds.append(f"{pname}=({vs},0)")
        cmds.append(f'Tx{label}=Text("{label}",({v - 0.18},0.62))')
        vals.append(pname)
    return cmds, vals


# ---------------------------------------------------------------------------
# PIL 裁留白（治引擎纵横比校正把数轴拉成高图）
# ---------------------------------------------------------------------------
def trim_whitespace(png_path: str, *, border: int = 24) -> str | None:
    """裁掉四周纯白留白（ImageChops.difference 求内容 bbox + 留 border 边）。
    成功 → 写一个新 *_trim.png 路径并返回；失败/无内容 → None（上层用原图兜底）。"""
    try:
        from PIL import Image, ImageChops
    except Exception as e:  # noqa: BLE001 — PIL 不可用 → None（用原图）
        logger.info("numline trim：PIL 不可用 %s", str(e)[:80])
        return None
    try:
        im = Image.open(png_path).convert("RGB")
        # 背景取左上角像素（无头渲染为纯白）；求与背景的差异 bbox。
        bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if not bbox:
            return None  # 全白（无内容）→ 上层兜底
        left, top, right, bottom = bbox
        left = max(0, left - border)
        top = max(0, top - border)
        right = min(im.width, right + border)
        bottom = min(im.height, bottom + border)
        cropped = im.crop((left, top, right, bottom))
        out_path = os.path.splitext(png_path)[0] + "_trim.png"
        cropped.save(out_path, "PNG")  # PNG 无损（图全链铁律）
        return out_path
    except Exception as e:  # noqa: BLE001 — 裁剪异常 → None（用原图，绝不因裁剪崩）
        logger.info("numline trim 异常: %s", str(e)[:120])
        return None


# ---------------------------------------------------------------------------
# 入口：数轴确定性造图
# ---------------------------------------------------------------------------
async def compose_number_line_figure(
    *,
    stem: str,
    invoke: Any,
    parse_json: Any,
    item_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """数轴确定性造图。成功 → 与 compose_variant_figure 同格式的成功 dict（含 png_path / commands / vals）；
       任何降级信号（检测不到点 / 渲染失败 / 裁剪后无内容）→ 返回 None，**上层回退 LLM 造图路径**（绝不卡死）。

    🔴 返回 dict 时给 png_path（让上层统一转 base64 + 组装最终 result，避免本模块耦合 base64/result 格式）。
    """
    # 1) LLM 抽点（只抽结构化点值，不画图）
    extracted = await _extract_points(stem=stem, invoke=invoke, parse_json=parse_json, model=model)
    if not extracted:
        logger.info("numline item=%s 抽点失败/无已知点 → 回退 LLM 造图", item_id)
        return None  # 回退
    points = extracted["points"]
    vmin, vmax = extracted["min"], extracted["max"]

    # 2) 确定性模板 → 命令
    commands, vals_names = build_commands(points, vmin, vmax)

    # 3) 走管线现有 GeoGebra 渲染（axes/grid false, mono）。
    #    🔴 auto_labels=False：字母由独立 Text 放轴上方（确定性位置），不让引擎再在点旁印一遍内部点名
    #       （否则与 Text 字母双标，见用户验证 spec 的 autoLabels:false）。
    #    🔴 point_size 不传（走引擎默认）：默认渲染成干净实心圆点（对齐用户参考 numline_test1）；
    #       point_size=4 反而在 mono 下渲成箭头状（实测），故不覆盖。
    try:
        r = mathfig_render.render(
            commands,
            dashed=None, hide=None, vals=vals_names,
            axes=False, grid=False,
            stem=f"numline_{item_id or 'fig'}",
            fig_scale=None,        # 数轴模板坐标已定死，不缩画布
            point_size=None,       # 引擎默认 = 干净实心圆点（用户验证效果）
            relabel=None, angle_labels=None,
            auto_labels=False,     # 字母走独立 Text，不要引擎重印内部点名
        )
    except Exception as e:  # noqa: BLE001 — 渲染冒泡 → 回退
        logger.info("numline item=%s 渲染异常 %s → 回退 LLM 造图", item_id, str(e)[:120])
        return None
    if not r.get("ok") or not r.get("png_path"):
        logger.info("numline item=%s 渲染未成功(%s) → 回退 LLM 造图", item_id, r.get("error"))
        return None

    # 4) PIL 裁留白（引擎纵横比校正把数轴拉成高图）；裁失败用原图兜底。
    trimmed = trim_whitespace(r["png_path"])
    final_png = trimmed or r["png_path"]

    logger.info(
        "numline item=%s 确定性出图成功 points=%s range=[%d,%d] trimmed=%s",
        item_id, sorted(points.items()), vmin, vmax, bool(trimmed),
    )
    return {
        "ok": True,
        "png_path": final_png,
        "commands": commands,
        "vals": r.get("vals", {}),
        "warnings": r.get("warnings", []),
        "numline_deterministic": True,   # 命中观测：本图走了数轴确定性模板（非 LLM 手搓）
        "numline_points": {k: v for k, v in sorted(points.items())},
        "numline_range": [vmin, vmax],
    }
