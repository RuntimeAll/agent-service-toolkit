# -*- coding: utf-8 -*-
r"""PRD-C-100 B3 带图管线 · service 层后处理（D14：不进变式 StateGraph，四节点字节不动）。

三件事（人在回路）：
  B3.1 母题切图 `crop_mother_figure`：带图母题 → figure-crop 检测+裁 → PNG base64（母题图直贴）。
  B3.2 变式造图一轮直出 `compose_variant_figure`：opus 翻 GeoGebra 命令（不调 verify_figure 回炉 D12）
       → mathfig 渲染 → PNG base64。⚙模板（geogebra_samples）+ needs_figure 降级。
  B3.3 图片重生：同 compose_variant_figure 带 correction_prompt（老师发修正提示词 → 单图重造）。

🔴 铁律：图全链 PNG 无损；渲染失败 try/except 降级（needs_figure ⚠ 外显，不掐 SSE/不卡流程 G11）；
   opus 翻命令走 invoke(=_ainvoke_text) 落 conv_trace（label=figure_geogebra，G6 opus 调用不漏计）；
   画图链/图片重生**不挂缓存**（低频+每次变，开缓存更贵）。
🔴 OSS：本模块只产 PNG base64（人在回路展示，省 OSS spam）；入库时 FE 传 OSS（uploadMotherImage 既有）
   → A-015 image 块。本模块不耦合 OSS。
"""
from __future__ import annotations

import base64
import os
import tempfile
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from agents.figure import geogebra_samples, mathfig_render

# opus 翻 GeoGebra 命令 system 前缀（稳定，喂样例 + 命令语义坑）。画图链不挂缓存，但 prompt 仍分层清晰。
_GEO_SYSTEM = (
    "你是中小学数学配图助手。把给定的「变式题面 + 解答」翻译成一组 GeoGebra evalCommand 命令"
    "（每行一条，能被 GeoGebra Math Apps 执行渲染成题目配图）。\n"
    "🔴 强制工作流（必走三步，治「首次造图漏元素」——题面要标的角/记号别漏画）：\n"
    "  第一步·列清单：先逐一列出题面/解答里**明确提到的所有几何对象**——点、角"
    "（含「角1/角2/∠1/∠2」这类**要求标注**的角）、线段、辅助线、记号（直角/相等/平行/角度数）"
    "——形成元素清单（在脑中或注释里过一遍，不外放到 JSON）。\n"
    "  第二步·逐项落命令：保证清单里**每一项都有对应的 GeoGebra 命令或标注**"
    "（标注角用 Angle(...)，点序见下；要标的角必须有 Angle 命令，绝不只画线不标角）。\n"
    "  第三步·交付前自检：对照第一步的元素清单**逐项核对**，确认**无遗漏**"
    "（题面要的角/记号都画了）、也**无擅自新增**题面不存在的元素（忠实原图最小集——"
    "原图没有的圆点/角弧/直角小方块/辅助线一律不加）。\n"
    "🔴 命令语义坑（必守，防返工）：① 自由点直接 A=(2,3)，**别用 Point((2,3))**（会失败）；"
    "② 派生点（交点/垂足/中点/旋转像）用 Intersect/Midpoint/PerpendicularLine/Rotate/Reflect/Translate"
    "让引擎算，别手填坐标——垂足用 ClosestPoint(Line(A,E),B)（求 B 到 AE 的垂足）、"
    "三点定圆用 Circle(A,B,C)（过三点画圆）、圆心用 Center(c)、垂直平分线用 PerpendicularBisector(B,N)；"
    "③ **别在 commands 里写 SetLineStyle**（会把对象渲染没）——虚线走 dashed 字段；"
    "④ 变换题的「像」、立体隐藏棱放进 dashed；⑤ 关键点放进 vals 自核；"
    "⑥ 🔴 标注角的点序坑：Angle(P,V,Q) 是**有向角**（V 是顶点，从 V→P 逆时针扫到 V→Q），"
    "点序须让扫角 ≤180°，否则画成反向优角（曾把直角渲成 270° 大半圆）——"
    "标 ∠BAC 写 Angle(B,A,C)（顶点 A 在中间），扫出来大于平角就把首尾两点调换。\n"
    "🔴 只输出一个 JSON（不要解释、不要 markdown fence）：\n"
    '{"commands":["...","..."],"dashed":["对象名"],"hide":["辅助对象名"],"vals":["关键点名"],'
    '"axes":false,"needs_figure":false}\n'
    "🔴 若此题**不适合/不需要配图**（纯代数无几何意义、或你无法可靠构造），把 needs_figure 设 true、"
    "commands 留空数组（降级，不硬画错图）。\n\n"
    + geogebra_samples.samples_prompt_block()
)


def _png_to_b64(png_path: str) -> str | None:
    try:
        with open(png_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


async def compose_variant_figure(
    *,
    stem: str,
    answer: str | None = None,
    invoke: Any,
    parse_json: Any,
    correction_prompt: str | None = None,
    prev_commands: list[str] | None = None,
    item_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """变式造图一轮直出（+ 图片重生）。返回
       {item_id, ok, png_base64?, commands, dashed, vals, needs_figure, warnings, reason?}。

    🔴 invoke = variant._ainvoke_text（落 conv_trace label=figure_geogebra）；parse_json = variant._parse_json。
    🔴 correction_prompt 非空 = 图片重生（老师修正提示词，单图重造，人在回路 D12）。
    🔴 PRD-C-100 C：图片重生带上一版 GeoGebra commands（prev_commands）+ 原题上下文 →
       opus 在「上一版配图命令的基础上按修正要求调整」（增量修改，而非从零重画，保证继承上一版）。
       仅当 correction_prompt 与 prev_commands 同时非空才走增量分支；首次造图（无 prev_commands）维持原行为。
    🔴 任何失败 → needs_figure=True（降级，不抛、不掐流程 G11）。
    """
    # 🔴 B5 预算护栏（G7）：当日花费超阈值 → 造图降级（needs_figure，不调翻命令，题照常交付）。
    from agents import cost_guard
    # 🔴 P5：护栏读库丢线程池（async 版），慢库不卡 asyncio loop / 不拖垮并发 SSE。
    if await cost_guard.is_budget_exceeded_async():
        return {"item_id": item_id, "ok": False, "needs_figure": True,
                "reason": "今日 AI 额度已用尽，配图暂缓（可明日重试或手动配图）", "commands": [], "warnings": []}

    user_segs = [f"【变式题面】\n{stem}"]
    if answer:
        user_segs.append(f"【标准答案/解答】\n{answer}")
    # 🔴 PRD-C-100 C：图片重生 = correction_prompt + prev_commands 都在 → 增量修改（带上一版命令 + 原题上下文）。
    #   把上一版 GeoGebra commands 原样喂回，指令改成「在下面这版配图命令的基础上，按修正要求调整」，
    #   让 opus 继承上一版骨架做增量改动，而非丢掉上下文从零重画（跑偏的根因）。
    incremental_regen = bool(correction_prompt and prev_commands)
    if incremental_regen:
        prev_block = "\n".join(str(c) for c in (prev_commands or []) if str(c).strip())
        user_segs.append(f"【上一版配图 GeoGebra 命令（基准，在此之上调整）】\n{prev_block}")
        user_segs.append(
            "【老师修正要求（🔴 在上面这版配图命令的基础上，按本要求调整——而非从零重画，"
            f"保留与修正无关的部分、只改需要改的）】\n{correction_prompt}"
        )
    elif correction_prompt:  # 图片重生但无上一版命令（首版即降级 / 老会话）→ 退回原行为（从修正词重画）
        user_segs.append(f"【老师修正要求（按此重新构造配图）】\n{correction_prompt}")
    messages = [
        SystemMessage(content=_GEO_SYSTEM),
        HumanMessage(content="\n\n".join(user_segs)),
    ]
    # opus 翻命令（走 _ainvoke_text 落 conv_trace；label 由 system 头 "GeoGebra evalCommand" 决定，
    # B1b trace marker 已加 figure_geogebra 钩子见 variant._TRACE_MARKERS 同批补）。
    try:
        # model 默认 None → 调用方（service 端点）传 settings.VARIANT_MODEL_FIGURE（opus）；
        #   传了 model 才走 per-call 覆盖（温度 0.2，timeout 留默认；造图链不挂缓存）。
        text = await invoke(messages, model=model, max_tokens=4096, temperature=0.2)
    except Exception as e:  # noqa: BLE001 — opus 翻命令失败 → needs_figure 降级
        return {"item_id": item_id, "ok": False, "needs_figure": True,
                "reason": f"opus 翻命令失败: {str(e)[:80]}", "commands": [], "warnings": []}

    data = parse_json(text)
    if not isinstance(data, dict) or data.get("needs_figure") is True or not data.get("commands"):
        return {"item_id": item_id, "ok": False, "needs_figure": True,
                "reason": "opus 判定不适合配图或未给命令", "commands": (data or {}).get("commands", []),
                "warnings": []}

    r = mathfig_render.render(
        list(data.get("commands") or []),
        dashed=data.get("dashed"), hide=data.get("hide"), vals=data.get("vals"),
        axes=bool(data.get("axes")), stem=f"variant_{item_id or 'fig'}",
    )
    if not r.get("ok") or not r.get("png_path"):
        return {"item_id": item_id, "ok": False, "needs_figure": True,
                "reason": r.get("error") or "渲染失败", "commands": data.get("commands"),
                "warnings": r.get("warnings", [])}
    b64 = _png_to_b64(r["png_path"])
    if not b64:
        return {"item_id": item_id, "ok": False, "needs_figure": True,
                "reason": "PNG 读取失败", "commands": data.get("commands"), "warnings": []}
    return {
        "item_id": item_id, "ok": True, "needs_figure": False, "png_base64": b64,
        "commands": data.get("commands"), "dashed": data.get("dashed"),
        "vals": r.get("vals", {}), "warnings": r.get("warnings", []),
    }


async def crop_mother_figure(
    image_url: str, *, conf: float = 0.2,
) -> dict[str, Any]:
    """母题切图（B3.1）：下载母题图 → figure-crop 检测+裁 → 第一张 figure 的 PNG base64。
       返回 {ok, png_base64?, n_figures, bbox?, needs_figure, reason?}。
    🔴 纯文字/公式题（figures=[]）→ needs_figure=False + ok=False（母题无图，正常，不降级告警）。
    🔴 任何 IO/模型失败 → needs_figure=True（降级，不抛）。
    🔴 P4（B 家族·配图全覆盖，方案 b）：多图母题（n_figures>1）仍只回第 1 张切图（FE 母题展示单图），
       但 n_figures 记真实张数 + reason 外显「本题含 N 图、已切第 1 张，如缺图可手动重切」——
       绝不静默丢图（旧实现只取 figs[0]、n_figures 记了真实数量却无任何外显提示）。
       不改 (a) 回多图：母题原图本就直贴展示、多图母题罕见，回多 base64 + FE 多图渲染收益不抵风险。
    """
    tmp_path = None
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(resp.content)
        from agents.figure.figure_crop import detect_and_crop  # 进程内 import（torch 懒加载）
        det = detect_and_crop(tmp_path, conf=conf)
        figs = det.get("figures") or []
        if not figs:
            return {"ok": False, "needs_figure": False, "n_figures": 0,
                    "reason": "母题无图形（纯文字/公式题）"}
        first = figs[0]
        b64 = _png_to_b64(first["crop_path"])
        n = len(figs)
        if not b64:
            return {"ok": False, "needs_figure": True, "n_figures": n,
                    "reason": "切图 PNG 读取失败"}
        # P4：多图母题不静默丢图——回第 1 张 + reason 外显真实张数，提示老师可手动重切补图。
        multi_hint = (
            f"本题含 {n} 图，已切第 1 张；如缺图可手动重切" if n > 1 else None
        )
        return {"ok": True, "needs_figure": False, "n_figures": n,
                "png_base64": b64, "bbox": first.get("bbox"), "conf": first.get("conf"),
                "reason": multi_hint}
    except Exception as e:  # noqa: BLE001 — 下载/检测失败 → 降级
        return {"ok": False, "needs_figure": True, "reason": f"母题切图失败: {str(e)[:80]}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
