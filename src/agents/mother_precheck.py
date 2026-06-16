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
#
# 🔴 接地（2026-06-16）：旧 prompt「开放生成」让 nano 凭记忆自由写年级册名 / 章名，且默认按
#   人教版章序判 → 与库内**浙教版** biz_subject 命名口径对不上（老师上传后弹窗预填从不准）。
#   现把真实闭集喂进去让 nano「从给定列表里选」：
#   ① 年级册闭集 = biz_subject level1 的 6 个**教材册**（七上/七下/八上/八下/九上/九下，
#      实查 id 3071/3072/3081/3082/3091/3092）。复习/专题册（中考一轮复习 3010 / 数学解题技巧
#      与专题 3100 / 新题抢先 3120）**不作母题年级册**——母题是某一具体章的题。
#   ② 章口径 = 浙教版，章名用**中文数字**「第二章一元二次方程」（库内真实写法），不是阿拉伯
#      数字 + 空格「第2章 一元二次方程」。库内章名空格不统一（有的带空格、有的不带），nano
#      照常识写即可，下游按名匹配容差。
#   ❗章树本身（册→章 全列表 + id）本轮未注入（节点在 classify 前、无 ruoyi_token、注入需新加
#     一次重 lazyTree 往返，超出 prompt 基础修复范围）→ chapter 仍由 nano 凭浙教版常识写、老师
#     确认；真正锚 subject_id 仍在 classify 用确认章 id 完成。详见模块末注 + 交付报告。
GRADE_BOOK_CLOSED_SET = "七年级上册、七年级下册、八年级上册、八年级下册、九年级上册、九年级下册"

NANO_PRECHECK_PROMPT = """你是**浙教版**初中数学题库管理员。看这张题目图，判断它属于哪个**年级册 + 章**，给出候选，并判题面**是否含图形**。

🔴 年级册**必须从下面 6 个里选一个**（这是题库 biz_subject 的真实命名，"册"是教材分册、不是"初一/七年级上学期"这类口径），拿不准就留空串、别自造：
{grade_book_set}
（中考一轮复习 / 专题 / 新题抢先等复习专题册**不算**母题年级册——母题是某一具体章的题。）

🔴 章按**浙教版**章序判（题面常无版本标记，别按人教版章序，否则与题库浙教版章对不上）。章名用**中文数字**写，如「第二章一元二次方程」「第三章圆的基本性质」，**不要**写成阿拉伯数字「第2章」。

{grade_hint}

只输出一个 JSON（不要解释、不要 markdown fence）：
{{
  "grade_book": "从上面 6 个里选最可能的一个(如:八年级下册)；拿不准留空串",
  "chapter": "最可能的章名(浙教版·中文数字，如:第二章一元二次方程)；拿不准留空串",
  "grade_candidates": ["其它可能的年级册(仍须是上面 6 个之一)，0~3 个；没有就空数组"],
  "chapter_candidates": ["其它可能的章名(浙教版·中文数字)，0~3 个；没有就空数组"],
  "has_figure": true/false,
  "confidence": 0.0~1.0
}}
🔴 has_figure 判定（偏保守）：题面真含**几何图/函数图象/统计图/数轴/图表**才填 true；
   只是**拍照的纯文字题**（哪怕拍歪、有底纹/手写痕迹）填 false。拿不准 → 填 false（宁漏勿误杀正常题）。"""


def build_precheck_prompt(grade_text_hint: str | None = None) -> str:
    """组 nano 前置判 prompt。grade_text_hint = analyze 已读出的年级粗值（只作参考，不锁死）。

    🔴 注入 6 教材册闭集 + 浙教版/中文数字章口径（接地，见 NANO_PRECHECK_PROMPT 注）。
    """
    if grade_text_hint:
        hint = f"（参考：上游读图初判像「{grade_text_hint}」，仅供参考，以你看图为准——且年级册仍须落在上面 6 个里。）"
    else:
        hint = ""
    return NANO_PRECHECK_PROMPT.format(
        grade_book_set=GRADE_BOOK_CLOSED_SET, grade_hint=hint
    )


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
