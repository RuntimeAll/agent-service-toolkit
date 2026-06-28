# -*- coding: utf-8 -*-
r"""PRD-A-002 B4 · 批改（给学生作答说对错）。无状态、单题粒度、opus 多模态。

🔴 维护者拍板（2026-06-26）批改逻辑：
   - 输入 = **原图坐标裁剪**出的题区图（含印刷体原题 + 学生手写作答） + 注入的知识点/章节。
   - 内部流程 = **先解题**（把标准答案/解题结果全拿出来）→ **再据此判学生作答对错**。
   - 多异常态：① 无笔迹 → verdict=blank（未作答，不臆造对错）；② 原题无答案无解析 → 自己先解题；
     ③ 对错以严谨解题为准、存疑（解不确定）→ verdict=uncertain 转人工，绝不硬判。
   - 条件出现：仅识别到笔迹（needGrading=1）才在前端出「批改」按钮——本模块只管被调到时正确批改。

prompt 通用（feedback_prompt_general_no_noise）：角色+输入规范+输出规范，无单题型噪音；
   知识点/章节作为**上下文注入**帮助判定，不写成某题型特化指令。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agents import dna_extract
from agents.recognize import parse_json_lax, _normalize_image_ref

GRADE_TIMEOUT_S = 180.0
GRADE_TEMPERATURE = 0.1
GRADE_MAX_TOKENS = 4096


def build_grade_prompt(*, knowledge: str | None = None, chapter: str | None = None) -> str:
    ctx_lines = []
    if (chapter or "").strip():
        ctx_lines.append(f"所属章节：{chapter.strip()}")
    if (knowledge or "").strip():
        ctx_lines.append(f"相关知识点：{knowledge.strip()}")
    ctx = ("\n".join(ctx_lines) + "\n") if ctx_lines else ""
    return f"""你是经验丰富的数学老师，正在**批改学生作答**。这张图是一道题区照片：含【印刷体原题】+【学生手写作答】。
{ctx}🔴 批改流程（务必按序）：
1) 先把【印刷体原题】读出来（去掉手写，得到干净题面）。
2) **先自己严谨解题**，得到标准答案与解题过程（题目本身没给答案/解析也要自己解出来）。
3) 再读【学生手写作答】（最终答案 + 过程），**对照标准答案判对错**。
🔴 判定纪律：
- 只有真有手写作答才判对错；**图中无任何手写作答 → verdict="blank"**（未作答，不要编造）。
- 学生答案与标准答案一致 → "correct"；明确不一致 → "wrong"；过程对但最后算错/部分步骤对 → "partial"。
- 你自己解题都**不确定/存疑** → verdict="uncertain"（转人工），绝不硬判对错。
- feedback 给老师视角的简短点评（错在哪/扣分点/建议），中文，2~4 句，不堆砌。

================ 输出（只输出一个 JSON，无解释、无 markdown fence） ================
{{
  "stem": "印刷体原题(去手写, Markdown+行内$LaTeX$)",
  "qtype": "选择/填空/解答",
  "standard_answer": "你解出的标准答案",
  "standard_analysis": "标准解题过程(干净)",
  "student_answer": "学生手写的最终答案(无则空串)",
  "has_handwriting": true/false,
  "verdict": "correct/wrong/partial/blank/uncertain",
  "feedback": "老师视角点评"
}}"""


async def grade_one(
    *,
    image_url: str | None = None,
    image_base64: str | None = None,
    knowledge: str | None = None,
    chapter: str | None = None,
    invoke: Any,
) -> dict[str, Any]:
    """批改单道题区图。无状态。永不抛到端点外（异常收口 ok=False）。

    返回 {ok, stem, qtype, standard_answer, standard_analysis, student_answer,
          has_handwriting, verdict, feedback, verify, error}。
    """
    img = _normalize_image_ref(image_url, image_base64)
    prompt = build_grade_prompt(knowledge=knowledge, chapter=chapter)
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": img}},
    ])
    raw = await invoke(
        [msg],
        model="claude-opus-4-8",
        temperature=GRADE_TEMPERATURE,
        max_tokens=GRADE_MAX_TOKENS,
        timeout=GRADE_TIMEOUT_S,
    )
    data = parse_json_lax(raw)

    stem = str(data.get("stem") or "").strip()
    qtype = dna_extract._norm_qtype(data.get("qtype"))
    standard_answer = str(data.get("standard_answer") or "").strip()
    standard_analysis = str(data.get("standard_analysis") or "").strip()
    student_answer = str(data.get("student_answer") or "").strip()
    has_hw = bool(data.get("has_handwriting"))
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in ("correct", "wrong", "partial", "blank", "uncertain"):
        verdict = "uncertain"
    feedback = str(data.get("feedback") or "").strip()

    out: dict[str, Any] = {
        "ok": True,
        "stem": stem,
        "qtype": qtype,
        "standard_answer": standard_answer,
        "standard_analysis": standard_analysis,
        "student_answer": student_answer,
        "has_handwriting": has_hw,
        "verdict": verdict,
        "feedback": feedback,
        "verify": None,
        "error": None,
    }

    # R5 sympy 验算标准答案（给老师一道复核；存疑不臆造）
    if standard_answer and stem:
        try:
            from agents.variant import verify_one_stem
            out["verify"] = await verify_one_stem(stem, standard_answer, qtype=qtype)
        except Exception as e:  # noqa: BLE001
            out["verify"] = {"verdict": "degrade", "detail": f"验算异常: {str(e)[:80]}", "computed": None}

    return out
