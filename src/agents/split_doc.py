# -*- coding: utf-8 -*-
r"""PRD-A-002 路B · 批量拆题「/split」无状态核心（统一 opus，零外部 OCR，无 figure_crop）。

🔴 心智：book-server 程序化抽取（POI/PDFBox 文字层 → markdown；扫描/公式重 → 栅格化页图）后，
   把抽取产物丢给本模块，opus 一次做「切题 + 答案配对 + 选项归位 + 题型判别 + 完整度过滤」
   （B0 实测：这些是 LLM 强项、程序化死穴 9/9 vs 0/9）。无状态：不读会话/DB/不落库。

与 PRD-C-101 §3.1 拆题层一致，但本卡（PRD-A-002）R2/C6 **弃用 figure_crop/YOLO**：不切图，
   配图由 book-server 在录入时把来源页图随 has_figure 题挂上（原图无损 R7）。

答案模式（处理选项·答案解析）：
  from_source 原卷自带 → 从抽取产物抽每题答案/解析（没有就空）。
  ai_solve   AI解题   → opus 亲解每题 → 答案/解析 + 10 维 DNA（R1，复用 opus_to_dna 归一）。
  stem_only  只录题   → 只拆题干，答案/解析留空。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agents import dna_extract, mother_opus
from agents.recognize import parse_json_lax

SPLIT_TIMEOUT_S = 240.0          # 整卷拆题可能慢（多题 + 多模态）
SPLIT_TEMPERATURE = 0.1
SPLIT_MAX_TOKENS = 16000         # 整卷多题，给足输出头

ANSWER_MODES = ("from_source", "ai_solve", "stem_only")


def build_split_prompt(*, answer_mode: str, grade_hint: str | None = None) -> str:
    """组拆题 prompt（通用，无单题型噪音）。answer_mode 决定答案/DNA 段。"""
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    grade_line = f"学段提示：{grade_hint}（解法/考点不超本学段）。" if grade_hint else ""

    base = """你是数学试卷拆题专家。把下面这份试卷/教辅内容拆成一道道**独立题目**。
🔴 拆题铁则：
- 每道题独立成项；**母子题**（一大题含多个小问①②③）**整体作为一道题**，小问保留在该题 stem 里，绝不拆散成多题。
- 去除页眉页脚/页码/卷头说明/班级姓名栏/装饰性文字，只保留真题。
- stem 用 Markdown + 行内 $LaTeX$；选择题选项逐项进 options（**不含** "A." 前缀），题干不重复选项。
- qtype 题型：选择/填空/解答 之一。
- has_figure：该题含几何图/函数图/图表填 true，纯文字填 false。
- **完整度过滤**：残缺/被截断/识别不全的题**不硬凑**，丢进 dropped（每项给一句原因）。"""

    if answer_mode == "ai_solve":
        ans_block = f"""
答案模式 = AI解题：你**亲自把每道题解出来**（严谨解题、不跳步、不臆造），据解答填 answer/analysis 并产出 10 维 DNA。
- assessmentType 闭集10选1 = {exam_types}；difficulty 1~4；tags 3~6 禁近义；solutionSkeleton 最难一步用【】包住；hardPointCount 必等于 breakthroughPoints 长度；primaryKp 只给 name。"""
        q_shape = """{
      "stem": "题干markdown+$latex$", "qtype": "选择/填空/解答", "options": ["选项正文"],
      "has_figure": true/false, "answer": "标准答案", "analysis": "解析",
      "dna": {"primaryKp": {"name": "主考点"}, "secondaryKps": [], "qtype": "解答",
        "assessmentType": "闭集之一", "solutionSkeleton": ["步骤"], "hardPointCount": 0,
        "breakthroughPoints": [], "scenario": "场景", "difficulty": 1, "tags": ["标签"], "modelCandidates": []}
    }"""
    elif answer_mode == "from_source":
        ans_block = "\n答案模式 = 原卷自带：从本文档里**抽取**每题对应的标准答案/解析填 answer/analysis（文档没给就留空串，不要自己编）。"
        q_shape = '{"stem": "...", "qtype": "选择/填空/解答", "options": [], "has_figure": false, "answer": "原卷答案或空", "analysis": "原卷解析或空"}'
    else:  # stem_only
        ans_block = "\n答案模式 = 只录题：只拆题干，answer/analysis 一律留空串。"
        q_shape = '{"stem": "...", "qtype": "选择/填空/解答", "options": [], "has_figure": false, "answer": "", "analysis": ""}'

    return f"""{base}{ans_block}
{grade_line}

================ 输出（只输出一个 JSON，无解释、无 markdown fence） ================
{{
  "questions": [
    {q_shape}
  ],
  "dropped": ["残缺题原因"]
}}"""


def _normalize_question(raw: dict[str, Any], *, answer_mode: str) -> dict[str, Any]:
    """单题归一（题型闭集 / 选项清洗 / DNA 归一）。"""
    stem = str(raw.get("stem") or "").strip()
    item: dict[str, Any] = {
        "stem": stem,
        "qtype": dna_extract._norm_qtype(raw.get("qtype")),
        "options": [str(o).strip() for o in (raw.get("options") or []) if str(o).strip()],
        "has_figure": bool(raw.get("has_figure")),
        "answer": str(raw.get("answer") or "").strip(),
        "analysis": str(raw.get("analysis") or "").strip(),
        "dna": None,
    }
    if answer_mode == "ai_solve":
        try:
            item["dna"] = mother_opus.opus_to_dna(raw)
        except Exception:  # noqa: BLE001
            item["dna"] = None
    return item


async def split_doc(
    *,
    markdown: str | None = None,
    image_base64: Any = None,          # str | list[str]（多模态页图）
    image_url: Any = None,             # str | list[str]
    answer_mode: str = "from_source",
    grade_hint: str | None = None,
    min_chars: int = 12,
    invoke: Any,
) -> dict[str, Any]:
    """拆题主入口：opus 一次拆题（文本档 / 多模态页图）。无状态。永不抛到端点外。

    返回 {ok, questions:[...], dropped:[...], count, error}。
    """
    if answer_mode not in ANSWER_MODES:
        answer_mode = "from_source"
    prompt = build_split_prompt(answer_mode=answer_mode, grade_hint=grade_hint)

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    has_input = False
    md = (markdown or "").strip()
    if md:
        content.append({"type": "text", "text": f"\n================ 文档内容（文字层） ================\n{md}"})
        has_input = True
    # 多模态页图（单张或多张）
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
        return {"ok": False, "questions": [], "dropped": [], "count": 0,
                "error": f"拆题输出解析失败: {str(e)[:100]}"}

    questions = []
    for q in (data.get("questions") or []):
        if not isinstance(q, dict):
            continue
        item = _normalize_question(q, answer_mode=answer_mode)
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
