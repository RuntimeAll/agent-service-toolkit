# -*- coding: utf-8 -*-
r"""PRD-A-002 B2 · 单题打标（10 维 DNA）。无状态、单题粒度。

用于**有原卷答案/解析**的题（from_source）：不必重解，只据已有题面+答案+解析产 DNA。
   无答案的题走 solve.py（解题=自动打标）。二者皆单题粒度并发，永不整卷一次大 JSON。
   复用 mother_opus.opus_to_dna 归一。prompt 通用，无单题型噪音。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agents import dna_extract, mother_opus
from agents.recognize import parse_json_lax
from core.settings import settings

LABEL_TIMEOUT_S = 120.0
LABEL_TEMPERATURE = 0.1
LABEL_MAX_TOKENS = 2048          # 单题仅 DNA，2k 足够


def build_label_prompt() -> str:
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    return f"""你是数学命题分析专家。下面给你**一道题**及其标准答案/解析，请只产出 **10 维 DNA 打标**
（不必重新解题，据已给信息分析）。
🔴 输出纪律：
- assessmentType 考察类型：**闭集10选1** = {exam_types}。
- solutionSkeleton 解法骨架：步骤序列，最难一步用【】整步包住（至多一处）。
- hardPointCount **必须等于** breakthroughPoints 数组长度（基础/套公式题→空数组、计 0）。
- difficulty 难度 1~4：1 送分 / 2 常规多步 / 3 含1难点或证明探究 / 4 压轴多难点。
- tags 检索标签 3~6（禁近义增生）；primaryKp 主考点只给 name；secondaryKps 0~3 个。
- modelCandidates 解题模型候选名（真有可复用套路才给，简单题空数组）。

================ 输出（只输出一个 JSON，无解释、无 markdown fence） ================
{{
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


async def label_one(
    *,
    stem: str,
    options: list[str] | None = None,
    qtype: str | None = None,
    answer: str | None = None,
    analysis: str | None = None,
    invoke: Any,
) -> dict[str, Any]:
    """对单道题（已有答案/解析）打标。无状态。永不抛到端点外。

    返回 {ok, dna, error}。
    """
    stem = (stem or "").strip()
    if not stem:
        return {"ok": False, "dna": None, "error": "题干为空"}

    qtype_n = dna_extract._norm_qtype(qtype)
    opts = [str(o).strip() for o in (options or []) if str(o).strip()]
    parts = [f"题型：{qtype_n}", "题干：", stem]
    if opts:
        parts.append("选项：")
        for i, o in enumerate(opts):
            parts.append(f"{chr(65 + i)}. {o}")
    if (answer or "").strip():
        parts.append(f"标准答案：{answer.strip()}")
    if (analysis or "").strip():
        parts.append(f"解析：{analysis.strip()}")
    question = "\n".join(parts)

    raw = await invoke(
        [HumanMessage(content=f"{build_label_prompt()}\n\n================ 待打标题目 ================\n{question}")],
        model=settings.LLM_MODEL_HEAVY,
        temperature=LABEL_TEMPERATURE,
        max_tokens=LABEL_MAX_TOKENS,
        timeout=LABEL_TIMEOUT_S,
    )
    data = parse_json_lax(raw)
    try:
        dna = mother_opus.opus_to_dna(data)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "dna": None, "error": f"DNA 归一异常: {str(e)[:80]}"}
    return {"ok": bool(dna), "dna": dna, "error": None if dna else "未产出 DNA"}
