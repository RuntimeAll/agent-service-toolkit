# -*- coding: utf-8 -*-
r"""PRD-C-017 B2 · 母题 nano 前置判（年级册 + 章 + 带图）+ 聚合/复习章识别（M7）。

🔴 心智（B2）：母题是「一对多放大器」，锚错章 = 制造想消灭的跨章串题（L-03 critical）。
   故在 opus 解题打标（classify）**之前**插一步 nano(gpt-5.4-nano)：
   ① 读母题原图判「年级册 + 章」（+ 候选）→ 发 needConfirm，**无条件停**等老师确认；
   ② 顺手判题面**是否含图形/图表/几何图**（拍照的纯文本题不算）→ 含图 → reject 终止流程。

本模块 = 纯函数 + 一个 nano LLM 调用包装（invoke 注入，便于单测桩）：
  - NANO_PRECHECK_PROMPT / build_precheck_prompt：复用 B0 探针 nano prompt（判章 + 判带图）。
  - precheck_judge：nano 一次多模态调用 → 归一 dict（grade_book/chapter/has_figure/candidates/conf）。
  - is_aggregation_chapter_name（M7）：章名是否「中考一轮复习 / 期末专题 / 专题」类聚合章
    （跨章大杂烩，误锚=串题）→ 标记/排除，**不拿它当锚定范围**。

🔴 false-positive 防护（带图判定）：纯文本误判带图 = 误杀正常题，比漏判更糟。判定偏保守——
   nano has_figure 取它如实返回，调用方对低置信容缺（拿不准当纯文本放行，宁可漏个别带图让 opus 兜）。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

# nano 判章/判带图调用上限（纯判定，远快于 opus 解题；给思考型留头）。
PRECHECK_TIMEOUT_S = 60.0
PRECHECK_TEMPERATURE = 0.2

# 复用 B0 探针 NANO_CHAPTER_PROMPT（tools/c017_b0_probe.py）的判章 + 判带图 prompt。
# 加候选输出（grade_candidates/chapter_candidates）给老师手选；候选可空。
NANO_PRECHECK_PROMPT = """你是浙教版初中数学题库管理员。看这张题目图，判断它属于哪个**年级册 + 章**，给出候选，并判题面**是否含图形**。

{grade_hint}

只输出一个 JSON（不要解释、不要 markdown fence）：
{{
  "grade_book": "最可能的年级册(如:八年级下册)；拿不准留空串",
  "chapter": "最可能的章名(如:第2章 一元二次方程)；拿不准留空串",
  "grade_candidates": ["其它可能的年级册，0~3 个；没有就空数组"],
  "chapter_candidates": ["其它可能的章名，0~3 个；没有就空数组"],
  "has_figure": true/false,
  "confidence": 0.0~1.0
}}
🔴 has_figure 判定（偏保守）：题面真含**几何图/函数图象/统计图/数轴/图表**才填 true；
   只是**拍照的纯文字题**（哪怕拍歪、有底纹/手写痕迹）填 false。拿不准 → 填 false（宁漏勿误杀正常题）。"""


def build_precheck_prompt(grade_text_hint: str | None = None) -> str:
    """组 nano 前置判 prompt。grade_text_hint = analyze 已读出的年级粗值（只作参考，不锁死）。"""
    if grade_text_hint:
        hint = f"（参考：上游读图初判像「{grade_text_hint}」，仅供参考，以你看图为准。）"
    else:
        hint = ""
    return NANO_PRECHECK_PROMPT.format(grade_hint=hint)


async def precheck_judge(
    *,
    image_url: str,
    invoke: Any,
    model: str,
    grade_text_hint: str | None = None,
    parse_json: Any,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """nano 一次多模态前置判（年级册 + 章 + 带图）。返回归一 dict。

    🔴 invoke = variant._ainvoke_text（落 trace/conv_trace + 走 relay 池）；parse_json = variant._parse_json。
       超时/失败由调用方接住（前置判失败 → 仍走 needConfirm 让老师全手选，不卡死，不静默放行带图）。
    """
    prompt = build_precheck_prompt(grade_text_hint)
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])
    kw: dict[str, Any] = dict(
        model=model,
        temperature=PRECHECK_TEMPERATURE,
        timeout=PRECHECK_TIMEOUT_S,
    )
    if max_tokens and max_tokens > 0:
        kw["max_tokens"] = max_tokens
    text = await invoke([msg], **kw)
    data = parse_json(text)
    return normalize_precheck(data)


def normalize_precheck(data: Any) -> dict[str, Any]:
    """nano 原始输出 → 归一 {grade_book, chapter, grade_candidates, chapter_candidates,
    has_figure, confidence}。非 dict / 缺字段 → 安全兜底（候选空、has_figure 兜 False 偏保守）。

    🔴 has_figure 兜底 = False（false-positive 防护：解析不出来当纯文本放行，宁漏勿误杀）。纯函数可单测。
    """
    if not isinstance(data, dict):
        data = {}

    def _strlist(raw: Any) -> list[str]:
        return [str(x).strip() for x in (raw or []) if str(x or "").strip()]

    grade_book = str(data.get("grade_book") or "").strip()
    chapter = str(data.get("chapter") or "").strip()
    # has_figure：只有 nano 明确给 true 才算 true（其余/缺失/非布尔 → False，偏保守）
    has_figure = data.get("has_figure") is True
    try:
        conf = float(data.get("confidence"))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return {
        "grade_book": grade_book,
        "chapter": chapter,
        "grade_candidates": _strlist(data.get("grade_candidates")),
        "chapter_candidates": _strlist(data.get("chapter_candidates")),
        "has_figure": has_figure,
        "confidence": conf,
    }


# ---------------------------------------------------------------------------
# M7 · 聚合/复习章识别（章名模式 + 非教学章黑名单）
# 实查 biz_subject：册 3071 下 3071007「中考一轮复习」/ 3071008「期末专题」= 跨章大杂烩，
# 叶子是跨多章题的混合，拿它当锚定范围 = 制造跨章串题。这类「聚合章」须标记/排除。
# ---------------------------------------------------------------------------
# 聚合章名关键词（命中任一 = 聚合/复习/专题，非单一教学章）。
_AGG_CHAPTER_RE = re.compile(
    r"中考|复习|专题|期末|期中|模考|一模|二模|三模|综合|总复习|压轴|真题|汇编|训练"
)


def is_aggregation_chapter_name(name: Any) -> bool:
    """章名是否「中考一轮复习 / 期末专题 / 专题」类聚合章（跨章大杂烩 → 误锚=串题，须排除）。

    🔴 M7：按 level2 章**名**模式判（_is_review_book 只判 4 位册前缀，拦不住册内聚合章）。
    纯函数可单测。
    """
    s = str(name or "").strip()
    if not s:
        return False
    return bool(_AGG_CHAPTER_RE.search(s))
