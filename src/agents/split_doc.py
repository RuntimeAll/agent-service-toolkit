# -*- coding: utf-8 -*-
r"""PRD-A-002 路B · 批量拆题「/split」无状态核心（**只拆题干 + 抽原卷解析，不解题不打标**）。

🔴 维护者拍板（2026-06-26）B2 根因纠正：旧实现把「拆题 + 逐题AI解题 + 10维打标」塞进**一次大
   opus JSON 调用** → 大卷输出（每题 500~1000 token × 数十题）超 max_tokens → 截断 → JSON 不闭合
   崩（"unbalanced JSON braces"）。修正 = **拆题/解题/打标三层分离**：
   - 本模块（split）：整页富文本（TextIn 转）→ **只切单题**（题干+选项+has_figure，from_source 时
     额外抽原卷答案/解析）。输出小、稳，大卷不再截断。
   - 解题（solve.py）/ 打标（label.py）：**单题粒度**，由 book-server worker 拆完后**并发逐题**调。

整页抽取（路B）走 TextIn 外部 OCR（textin_ocr.py，放宽 PRD-C-101 R3，仅限路B 整页）；
   本模块吃 markdown 文字层为主，image 仅作 TextIn 不可用时的 opus 多模态兜底。

答案模式：
  from_source 原卷自带 → 从富文本抽每题答案/解析（没有就空）。
  stem_only   只拆题干 → 答案/解析留空（解题留给下游 solve.py）。
  （旧 ai_solve 已废：解题下沉到单题 solve.py，split 不再在大调用里解题。传 ai_solve 按 stem_only 处理。）
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from agents import dna_extract
from agents.recognize import parse_json_lax

SPLIT_TIMEOUT_S = 240.0
SPLIT_TEMPERATURE = 0.1
# 🔴 整卷（一份完整试卷 ~25 题，含解答题多小问长 stem）只拆题干仍可达 ~2~3 万 token。
#   16000 实测整卷 2025杭州二模仍截断（unbalanced JSON braces）→ 提到 32000；
#   并配 _salvage_questions 截断兜底（超 32000 也只丢最后一题，不整单失败）。
SPLIT_MAX_TOKENS = 32000

# 兼容旧入参：ai_solve 不再在 split 里解题（下沉单题），按 stem_only 处理。
ANSWER_MODES = ("from_source", "stem_only", "ai_solve")


def build_split_prompt(*, answer_mode: str) -> str:
    """组拆题 prompt（通用，无单题型噪音，**不解题不打标**）。"""
    base = """你是数学试卷拆题专家。把下面这份试卷/教辅内容拆成一道道**独立题目**（**只拆题，不要解题**）。
🔴 拆题铁则：
- 每道题独立成项；**母子题**（一大题含多个小问①②③）**整体作为一道题**，小问保留在该题 stem 里，绝不拆散成多题。
- 去除页眉页脚/页码/卷头说明/班级姓名栏/装饰性文字，只保留真题。
- stem 用 Markdown + 行内 $LaTeX$；选择题选项逐项进 options（**不含** "A." 前缀），题干不重复选项。
- qtype 题型：选择/填空/解答 之一。
- has_figure：该题含几何图/函数图/图表填 true，纯文字填 false。
- **完整度过滤**：残缺/被截断/识别不全的题**不硬凑**，丢进 dropped（每项给一句原因）。"""

    if answer_mode == "from_source":
        ans_block = "\n答案模式 = 原卷自带：从本文档里**抽取**每题对应的标准答案/解析填 answer/analysis（文档没给就留空串，**不要自己解题/编造**）。"
    else:  # stem_only（含旧 ai_solve）
        ans_block = "\n答案模式 = 只拆题：answer/analysis 一律留空串（解题在下游单题进行，**此处绝不解题**）。"

    q_shape = '{"stem": "...", "qtype": "选择/填空/解答", "options": [], "has_figure": false, "answer": "原卷答案或空", "analysis": "原卷解析或空"}'

    return f"""{base}{ans_block}

================ 输出（只输出一个 JSON，无解释、无 markdown fence） ================
{{
  "questions": [
    {q_shape}
  ],
  "dropped": ["残缺题原因"]
}}"""


def _salvage_questions(raw_text: str) -> dict[str, Any] | None:
    """截断兜底：JSON 整体解析失败时，从 "questions":[ ... 里**逐个抠出完整题对象**，
    容忍被截断的最后一题（丢弃它而非整单失败）。对齐 B2 修复方向#3。

    返回 {"questions":[...], "dropped":[...]} 或 None（连一个完整题都抠不出）。
    """
    s = raw_text or ""
    s = re.sub(r"^```(?:json)?\s*", "", s.strip())
    m = re.search(r'"questions"\s*:\s*\[', s)
    if not m:
        return None
    i = m.end()  # 指向数组内第一个字符
    questions: list[dict[str, Any]] = []
    n = len(s)
    while i < n:
        # 跳到下一个 '{'（题对象开头）
        while i < n and s[i] not in "{]":
            i += 1
        if i >= n or s[i] == "]":
            break
        # 栈匹配抠出一个完整 {...}
        depth = 0
        in_str = False
        esc = False
        start = i
        end = -1
        while i < n:
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    i += 1
                    break
            i += 1
        if end < 0:
            break  # 最后一题被截断，丢弃
        try:
            questions.append(json.loads(s[start : end + 1]))
        except Exception:  # noqa: BLE001
            break
    if not questions:
        return None
    return {"questions": questions, "dropped": ["输出过长被截断，已保留前 %d 道完整题" % len(questions)]}


def _normalize_question(raw: dict[str, Any]) -> dict[str, Any]:
    """单题归一（题型闭集 / 选项清洗）。**不再产 DNA**（打标下沉单题 label.py）。"""
    stem = str(raw.get("stem") or "").strip()
    return {
        "stem": stem,
        "qtype": dna_extract._norm_qtype(raw.get("qtype")),
        "options": [str(o).strip() for o in (raw.get("options") or []) if str(o).strip()],
        "has_figure": bool(raw.get("has_figure")),
        "answer": str(raw.get("answer") or "").strip(),
        "analysis": str(raw.get("analysis") or "").strip(),
        "dna": None,                 # 占位：拆题阶段不打标，下游 label/solve 回填
    }


async def split_doc(
    *,
    markdown: str | None = None,
    image_base64: Any = None,          # str | list[str]（TextIn 不可用时的 opus 兜底页图）
    image_url: Any = None,             # str | list[str]
    answer_mode: str = "from_source",
    grade_hint: str | None = None,     # 兼容旧入参，本模块不再用（解法约束下沉 solve）
    min_chars: int = 12,
    invoke: Any,
) -> dict[str, Any]:
    """拆题主入口：opus 一次**只拆题**（不解题）。无状态。永不抛到端点外。

    返回 {ok, questions:[...], dropped:[...], count, error}。
    """
    if answer_mode not in ANSWER_MODES:
        answer_mode = "from_source"
    prompt = build_split_prompt(answer_mode=answer_mode)

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    has_input = False
    md = (markdown or "").strip()
    if md:
        content.append({"type": "text", "text": f"\n================ 文档内容（富文本/文字层） ================\n{md}"})
        has_input = True
    imgs: list[str] = []
    for ref in (image_url if isinstance(image_url, list) else [image_url] if image_url else []):
        if ref:
            imgs.append(str(ref).strip())
    for b in (image_base64 if isinstance(image_base64, list) else [image_base64] if image_base64 else []):
        if b:
            bs = str(b).strip()
            imgs.append(bs if bs.startswith("data:") else f"data:image/png;base64,{bs}")
    for u in imgs:
        content.append({"type": "image_url", "image_url": {"url": u}})
        has_input = True

    if not has_input:
        return {"ok": False, "questions": [], "dropped": [], "count": 0,
                "error": "markdown / image 至少给一个"}

    raw = await invoke(
        [HumanMessage(content=content)],
        model="claude-opus-4-8",
        temperature=SPLIT_TEMPERATURE,
        max_tokens=SPLIT_MAX_TOKENS,
        timeout=SPLIT_TIMEOUT_S,
    )
    try:
        data = parse_json_lax(raw)
    except Exception as e:  # noqa: BLE001
        # 🔴 截断兜底：整体 JSON 解析失败（多为输出超 max_tokens 截断）→ 抠出已完整的题，不整单失败
        data = _salvage_questions(raw)
        if not data:
            return {"ok": False, "questions": [], "dropped": [], "count": 0,
                    "error": f"拆题输出解析失败: {str(e)[:100]}"}

    questions = []
    for q in (data.get("questions") or []):
        if not isinstance(q, dict):
            continue
        item = _normalize_question(q)
        if len(item["stem"]) >= min_chars:
            questions.append(item)
    dropped = [str(d) for d in (data.get("dropped") or []) if str(d).strip()]

    return {
        "ok": True,
        "questions": questions,
        "dropped": dropped,
        "count": len(questions),
        "error": None if questions else "未拆出任何完整题目（请确认文件内容）",
    }
