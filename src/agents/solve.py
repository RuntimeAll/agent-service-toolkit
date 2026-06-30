# -*- coding: utf-8 -*-
r"""PRD-A-002 B2 · 单题解题（+ 自动打标 + sympy 验算）。无状态、单题粒度。

🔴 维护者拍板（2026-06-26）根因纠正：旧 split 把「整卷拆题 + 逐题解题 + 打标」塞进一次大
   opus JSON 调用 → 大卷输出超 max_tokens 截断 → JSON 不闭合崩。修正 = **拆题/解题/打标三层
   分离、单题粒度并发**：split 只拆题干（小输出稳）；本模块对**单道题**解题，由 book-server
   worker 并发逐题调（每次输出小，永不截断）。

解题=自动打标（设计稿 B4「解题 = AI 解这道题（自动打标）」）：一次产 答案/解析 + 10 维 DNA。
   复用 mother_opus.opus_to_dna 归一 + variant.verify_one_stem(R5 sympy 验算)。
   prompt 通用（feedback_prompt_general_no_noise）：角色+输入规范+输出规范，无单题型噪音。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agents import dna_extract, mother_opus
from agents.recognize import parse_json_lax
from core.settings import settings

SOLVE_TIMEOUT_S = 180.0
SOLVE_TEMPERATURE = 0.1
SOLVE_MAX_TOKENS = 4096          # 单题：题面+答案+解析+DNA，4k 足够，绝无整卷截断风险


def build_solve_prompt(*, grade_hint: str | None = None) -> str:
    """单题解题 + 10 维打标 prompt（通用）。grade_hint 缺省不约束。"""
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    grade_line = f"学段提示：{grade_hint}（解法不超出本学段进度）。" if grade_hint else ""
    return f"""你是严谨的数学解题与命题分析专家。下面给你**一道题**（题干 + 可能的选项），请：
1) **真正把它解出来**（一步步算到最终答案，不跳步、不臆造、不为凑预设答案反复改；信任自己的正确推导，与直觉不符最多复核一遍）；
2) 据你的解答产出 **10 维 DNA 打标**。
{grade_line}
🔴 输出纪律：
- answer = 标准答案（简短）；analysis = 干净最终解法（不写草稿/试错）；solvedAnswer = 你解出的最终答案（简短）。
- assessmentType 考察类型：**闭集10选1** = {exam_types}。
- solutionSkeleton 解法骨架：步骤序列，最难一步用【】整步包住（至多一处）。
- hardPointCount **必须等于** breakthroughPoints 数组长度（基础/套公式题→空数组、计 0）。
- difficulty 难度 1~4：1 送分 / 2 常规多步 / 3 含1难点或证明探究 / 4 压轴多难点。
- tags 检索标签 3~6（禁近义增生）；primaryKp 主考点只给 name；secondaryKps 0~3 个。
- modelCandidates 解题模型候选名（真有可复用套路才给，简单题空数组）。

================ 输出（只输出一个 JSON，无解释、无 markdown fence） ================
{{
  "answer": "标准答案",
  "analysis": "解析(干净最终解法)",
  "solvedAnswer": "你解出的最终答案(简短)",
  "dna": {{
    "primaryKp": {{"name": "主考点名"}},
    "secondaryKps": [{{"name": "副考点名"}}],
    "qtype": "选择/填空/解答",
    "assessmentType": "上述闭集10之一",
    "solutionSkeleton": ["步骤1", "步骤2"],
    "hardPointCount": 0,
    "breakthroughPoints": [],
    "scenario": "一句话场景 或 纯代数",
    "difficulty": 1,
    "tags": ["检索标签"],
    "modelCandidates": []
  }}
}}"""


def _build_question_text(stem: str, options: list[str] | None, qtype: str | None) -> str:
    lines = [f"题型：{qtype or '解答'}", "题干：", stem]
    if options:
        lines.append("选项：")
        for i, o in enumerate(options):
            lines.append(f"{chr(65 + i)}. {o}")
    return "\n".join(lines)


async def solve_one(
    *,
    stem: str,
    options: list[str] | None = None,
    qtype: str | None = None,
    grade_hint: str | None = None,
    invoke: Any,
) -> dict[str, Any]:
    """对单道题解题 + 打标 + 验算。无状态。永不抛到端点外（异常收口 ok=False）。

    返回 {ok, answer, analysis, solved_answer, dna, verify, richtext_issues, error}。
    """
    stem = (stem or "").strip()
    if not stem:
        return {"ok": False, "answer": "", "analysis": "", "solved_answer": "",
                "dna": None, "verify": None, "richtext_issues": [], "error": "题干为空"}

    qtype_n = dna_extract._norm_qtype(qtype)
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]
    prompt = build_solve_prompt(grade_hint=grade_hint)
    question = _build_question_text(stem, opts, qtype_n)

    raw = await invoke(
        [HumanMessage(content=f"{prompt}\n\n================ 待解题目 ================\n{question}")],
        model=settings.LLM_MODEL_HEAVY,
        temperature=SOLVE_TEMPERATURE,
        max_tokens=SOLVE_MAX_TOKENS,
        timeout=SOLVE_TIMEOUT_S,
    )
    data = parse_json_lax(raw)

    answer = str(data.get("answer") or "").strip()
    analysis = str(data.get("analysis") or "").strip()
    solved_answer = str(data.get("solvedAnswer") or "").strip()

    out: dict[str, Any] = {
        "ok": bool(answer or analysis),
        "answer": answer,
        "analysis": analysis,
        "solved_answer": solved_answer,
        "dna": None,
        "verify": None,
        "richtext_issues": [],
        "error": None,
    }

    # DNA 归一（自动打标，复用 mother_opus.opus_to_dna）
    try:
        out["dna"] = mother_opus.opus_to_dna(data)
    except Exception as e:  # noqa: BLE001
        out["dna"] = None
        out["error"] = f"DNA 归一异常: {str(e)[:80]}"

    # 富文本机器校验
    try:
        rv = mother_opus.validate_rich_text({"stem": stem, "answer": answer, "analysis": analysis})
        out["richtext_issues"] = rv.get("issues") or []
    except Exception:  # noqa: BLE001
        out["richtext_issues"] = []

    # R5 sympy 验算
    if answer:
        try:
            from agents.variant import verify_one_stem
            out["verify"] = await verify_one_stem(stem, answer, qtype=qtype_n, options=opts or None)
        except Exception as e:  # noqa: BLE001
            out["verify"] = {"verdict": "degrade", "detail": f"验算异常: {str(e)[:80]}", "computed": None}

    return out
